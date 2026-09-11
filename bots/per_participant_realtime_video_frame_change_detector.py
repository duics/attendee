import cv2
import numpy as np

# Thumbnail size used to compare frames
FINGERPRINT_WIDTH = 32
FINGERPRINT_HEIGHT = 18

# Mean absolute grayscale difference (0-255) below which two frames count as the same picture
FRAME_CHANGE_THRESHOLD = 2.0


def compute_frame_fingerprint(bgr_frame: np.ndarray) -> np.ndarray:
    """Small grayscale thumbnail of a BGR frame."""
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (FINGERPRINT_WIDTH, FINGERPRINT_HEIGHT), interpolation=cv2.INTER_AREA).astype(np.float32)


def frame_fingerprint_changed(previous_fingerprint: np.ndarray | None, fingerprint: np.ndarray, threshold: float = FRAME_CHANGE_THRESHOLD) -> bool:
    """True when the fingerprint changed enough to send the frame; no previous fingerprint counts as a change."""
    if previous_fingerprint is None or previous_fingerprint.shape != fingerprint.shape:
        return True
    return float(np.mean(np.abs(fingerprint - previous_fingerprint))) >= threshold


class PerParticipantRealtimeVideoFrameChangeDetector:
    """Compares each sampled frame with the last frame sent for one participant + source."""

    def __init__(self, threshold: float = FRAME_CHANGE_THRESHOLD):
        self.threshold = threshold
        self._last_sent_fingerprint = None
        self._candidate_fingerprint = None

    def frame_changed(self, bgr_frame: np.ndarray) -> bool:
        self._candidate_fingerprint = compute_frame_fingerprint(bgr_frame)
        return frame_fingerprint_changed(self._last_sent_fingerprint, self._candidate_fingerprint, self.threshold)

    def mark_sent(self):
        self._last_sent_fingerprint = self._candidate_fingerprint
