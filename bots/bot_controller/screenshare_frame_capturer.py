import base64
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

# dHash grid size: 8 rows x 9 columns give 64 adjacent-pixel comparisons
DHASH_SIZE = 8
# How many recently kept frames a new frame is compared against
RECENT_KEPT_FRAMES_TO_COMPARE = 5
# Hamming distance (bits out of 64) above which a frame counts as new
MIN_HAMMING_DISTANCE = 8
# Screenshare resolution requested while capturing frames
SCREENSHARE_FRAME_CAPTURE_RESOLUTION = "1080p"


def compute_dhash(image_bgr: np.ndarray) -> int:
    """64-bit difference hash of a BGR image: grayscale, shrink to 9x8, compare each pixel with its right neighbour."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (DHASH_SIZE + 1, DHASH_SIZE), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return int.from_bytes(np.packbits(diff.flatten()).tobytes(), byteorder="big")


def hamming_distance(hash_a: int, hash_b: int) -> int:
    return (hash_a ^ hash_b).bit_count()


def dhash_to_hex(dhash: int) -> str:
    return format(dhash, "016x")


@dataclass(frozen=True)
class KeptScreenshareFrame:
    jpeg_bytes: bytes
    participant_uuid: str
    # Milliseconds since the Unix epoch, same clock as utterances
    timestamp_ms: int
    dhash: int
    width: int
    height: int


class ScreenshareFrameCapturer:
    """
    Keeps a screenshare frame when its dHash differs enough from the last few kept frames for that participant.
    on_frame_kept is called on the thread that delivered the frame.
    """

    def __init__(
        self,
        on_frame_kept: Callable[[KeptScreenshareFrame], None],
        get_recording_start_timestamp_ms_callback: Callable[[], Optional[int]],
        logger: Optional[logging.Logger] = None,
    ):
        self._on_frame_kept = on_frame_kept
        self._get_recording_start_timestamp_ms = get_recording_start_timestamp_ms_callback
        self.log = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._recent_hashes_by_participant = defaultdict(lambda: deque(maxlen=RECENT_KEPT_FRAMES_TO_COMPARE))

    def add_frame(self, frame: bytes, participant_uuid: str, source: str):
        if source != "screenshare":
            return

        recording_start_timestamp_ms = self._get_recording_start_timestamp_ms()
        if recording_start_timestamp_ms is None:
            return

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms < recording_start_timestamp_ms:
            return

        try:
            jpeg_bytes = base64.b64decode(frame, validate=True)
        except Exception:
            self.log.warning("Screenshare frame for participant %s is not valid base64, skipping", participant_uuid)
            return

        image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.log.warning("Screenshare frame for participant %s could not be decoded as JPEG, skipping", participant_uuid)
            return

        dhash = compute_dhash(image)

        if not self._is_distinct_from_recent_kept_frames(participant_uuid, dhash):
            return

        height, width = image.shape[:2]
        self._on_frame_kept(
            KeptScreenshareFrame(
                jpeg_bytes=jpeg_bytes,
                participant_uuid=participant_uuid,
                timestamp_ms=timestamp_ms,
                dhash=dhash,
                width=width,
                height=height,
            )
        )

    def _is_distinct_from_recent_kept_frames(self, participant_uuid: str, dhash: int) -> bool:
        with self._lock:
            recent_hashes = self._recent_hashes_by_participant[participant_uuid]
            if any(hamming_distance(dhash, recent_hash) <= MIN_HAMMING_DISTANCE for recent_hash in recent_hashes):
                return False
            recent_hashes.append(dhash)
            return True
