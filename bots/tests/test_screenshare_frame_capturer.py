import base64
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from bots.bot_controller.screenshare_frame_capturer import (
    MIN_HAMMING_DISTANCE,
    RECENT_KEPT_FRAMES_TO_COMPARE,
    KeptScreenshareFrame,
    ScreenshareFrameCapturer,
    compute_dhash,
    dhash_to_hex,
    hamming_distance,
)

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
# Intensity step between neighbouring blocks. Large enough to survive JPEG compression and small noise.
BLOCK_STEP = 15


def make_image_with_dhash(dhash: int, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> np.ndarray:
    """
    Builds a BGR image whose difference hash is `dhash`.

    Each of the 8 rows starts at mid gray and steps up or down by BLOCK_STEP for each of the 8 bits
    of that row, giving a 9x8 grid of blocks that is scaled up to the requested size.
    """
    blocks = np.zeros((8, 9), dtype=np.uint8)
    for row in range(8):
        value = 128
        blocks[row, 0] = value
        for col in range(8):
            bit = (dhash >> (63 - (row * 8 + col))) & 1
            value = value + BLOCK_STEP if bit else value - BLOCK_STEP
            blocks[row, col + 1] = value
    gray = cv2.resize(blocks, (width, height), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def encode_frame(image_bgr: np.ndarray, jpeg_quality: int = 50) -> bytes:
    """Encodes an image the way the per-participant video path delivers it: base64 text of a JPEG."""
    ok, jpeg = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    assert ok
    return base64.b64encode(jpeg.tobytes())


def flip_low_bits(dhash: int, number_of_bits: int) -> int:
    return dhash ^ ((1 << number_of_bits) - 1)


HASH_A = 0xA5A5A5A5A5A5A5A5
HASH_B = 0x0F0F0F0F0F0F0F0F
HASH_C = 0x3C3C3C3C3C3C3C3C


class TestDhashFunctions(unittest.TestCase):
    def test_hamming_distance(self):
        self.assertEqual(hamming_distance(0, 0), 0)
        self.assertEqual(hamming_distance(0b1011, 0b0010), 2)
        self.assertEqual(hamming_distance(0, (1 << 64) - 1), 64)
        self.assertEqual(hamming_distance(HASH_A, HASH_A), 0)

    def test_dhash_to_hex(self):
        self.assertEqual(dhash_to_hex(0), "0" * 16)
        self.assertEqual(dhash_to_hex((1 << 64) - 1), "f" * 16)
        self.assertEqual(dhash_to_hex(HASH_A), "a5a5a5a5a5a5a5a5")

    def test_compute_dhash_is_64_bits(self):
        dhash = compute_dhash(make_image_with_dhash(HASH_A))
        self.assertIsInstance(dhash, int)
        self.assertGreaterEqual(dhash, 0)
        self.assertLess(dhash, 1 << 64)

    def test_compute_dhash_recovers_the_pattern_the_image_was_built_from(self):
        for expected in (0, (1 << 64) - 1, HASH_A, HASH_B, HASH_C):
            self.assertEqual(compute_dhash(make_image_with_dhash(expected)), expected)

    def test_compute_dhash_survives_jpeg_round_trip(self):
        for expected in (HASH_A, HASH_B):
            jpeg_bytes = base64.b64decode(encode_frame(make_image_with_dhash(expected)))
            decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(compute_dhash(decoded), expected)

    def test_compute_dhash_is_stable_under_small_noise(self):
        image = make_image_with_dhash(HASH_A)
        rng = np.random.default_rng(0)
        noise = rng.integers(-3, 4, size=image.shape, dtype=np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        self.assertLessEqual(hamming_distance(compute_dhash(image), compute_dhash(noisy)), MIN_HAMMING_DISTANCE)

    def test_compute_dhash_ignores_uniform_colour_changes(self):
        # A frame that is one flat colour has no edges, so every flat frame hashes to zero
        black = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        white = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 255, dtype=np.uint8)
        self.assertEqual(compute_dhash(black), 0)
        self.assertEqual(compute_dhash(white), 0)


class TestScreenshareFrameCapturer(unittest.TestCase):
    RECORDING_START_MS = 1_700_000_000_000

    def setUp(self):
        self.on_frame_kept = Mock()
        self.recording_start_timestamp_ms = self.RECORDING_START_MS
        self.capturer = ScreenshareFrameCapturer(
            on_frame_kept=self.on_frame_kept,
            get_recording_start_timestamp_ms_callback=lambda: self.recording_start_timestamp_ms,
        )
        self.now_seconds = (self.RECORDING_START_MS + 5_000) / 1000
        time_patcher = patch("bots.bot_controller.screenshare_frame_capturer.time")
        self.mock_time = time_patcher.start()
        self.addCleanup(time_patcher.stop)
        self.mock_time.time.side_effect = lambda: self.now_seconds

    def add_frame(self, dhash: int, participant_uuid: str = "participant_1", source: str = "screenshare"):
        self.capturer.add_frame(frame=encode_frame(make_image_with_dhash(dhash)), participant_uuid=participant_uuid, source=source)

    def kept_frames(self):
        return [call.args[0] for call in self.on_frame_kept.call_args_list]

    def test_first_screenshare_frame_is_kept_with_its_metadata(self):
        frame = encode_frame(make_image_with_dhash(HASH_A))
        self.capturer.add_frame(frame=frame, participant_uuid="participant_1", source="screenshare")

        self.on_frame_kept.assert_called_once()
        kept = self.kept_frames()[0]
        self.assertIsInstance(kept, KeptScreenshareFrame)
        self.assertEqual(kept.participant_uuid, "participant_1")
        self.assertEqual(kept.timestamp_ms, self.RECORDING_START_MS + 5_000)
        self.assertEqual(kept.dhash, HASH_A)
        self.assertEqual(kept.width, FRAME_WIDTH)
        self.assertEqual(kept.height, FRAME_HEIGHT)
        self.assertEqual(kept.jpeg_bytes, base64.b64decode(frame))

    def test_webcam_frames_are_ignored(self):
        self.add_frame(HASH_A, source="webcam")
        self.on_frame_kept.assert_not_called()

    def test_frames_are_skipped_until_the_recording_has_started(self):
        self.recording_start_timestamp_ms = None
        self.add_frame(HASH_A)
        self.on_frame_kept.assert_not_called()

        self.recording_start_timestamp_ms = self.RECORDING_START_MS
        self.add_frame(HASH_A)
        self.on_frame_kept.assert_called_once()

    def test_frames_that_arrive_before_the_recording_start_are_skipped(self):
        self.now_seconds = (self.RECORDING_START_MS - 1) / 1000
        self.add_frame(HASH_A)
        self.on_frame_kept.assert_not_called()

        self.now_seconds = self.RECORDING_START_MS / 1000
        self.add_frame(HASH_A)
        self.on_frame_kept.assert_called_once()

    def test_identical_frame_is_skipped(self):
        self.add_frame(HASH_A)
        self.add_frame(HASH_A)
        self.assertEqual(self.on_frame_kept.call_count, 1)

    def test_frame_within_the_hamming_threshold_is_skipped(self):
        self.add_frame(HASH_A)
        self.add_frame(flip_low_bits(HASH_A, MIN_HAMMING_DISTANCE))
        self.assertEqual(self.on_frame_kept.call_count, 1)

    def test_frame_just_beyond_the_hamming_threshold_is_kept(self):
        self.add_frame(HASH_A)
        self.add_frame(flip_low_bits(HASH_A, MIN_HAMMING_DISTANCE + 1))
        self.assertEqual(self.on_frame_kept.call_count, 2)

    def test_noisy_copy_of_a_kept_frame_is_skipped(self):
        image = make_image_with_dhash(HASH_A)
        rng = np.random.default_rng(1)
        noise = rng.integers(-3, 4, size=image.shape, dtype=np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        self.capturer.add_frame(frame=encode_frame(image), participant_uuid="participant_1", source="screenshare")
        self.capturer.add_frame(frame=encode_frame(noisy), participant_uuid="participant_1", source="screenshare")
        self.assertEqual(self.on_frame_kept.call_count, 1)

    def test_visibly_different_frames_are_all_kept(self):
        self.add_frame(HASH_A)
        self.add_frame(HASH_B)
        self.add_frame(HASH_C)
        self.assertEqual([kept.dhash for kept in self.kept_frames()], [HASH_A, HASH_B, HASH_C])

    def test_frame_is_compared_against_every_recent_kept_frame_not_just_the_last_one(self):
        self.add_frame(HASH_A)
        self.add_frame(HASH_B)
        # Going back to the first screen is a duplicate of a recently kept frame
        self.add_frame(HASH_A)
        self.assertEqual(self.on_frame_kept.call_count, 2)

    def test_frame_older_than_the_comparison_window_can_be_kept_again(self):
        distinct_hashes = [HASH_A] + [flip_low_bits(HASH_A, 16 + 12 * i) for i in range(RECENT_KEPT_FRAMES_TO_COMPARE)]
        for hash_x in distinct_hashes:
            for hash_y in distinct_hashes:
                if hash_x != hash_y:
                    self.assertGreater(hamming_distance(hash_x, hash_y), MIN_HAMMING_DISTANCE)

        # Fill the window with frames that all differ from HASH_A, pushing HASH_A out of it
        for dhash in distinct_hashes:
            self.add_frame(dhash)
        self.assertEqual(self.on_frame_kept.call_count, len(distinct_hashes))

        self.add_frame(HASH_A)
        self.assertEqual(self.on_frame_kept.call_count, len(distinct_hashes) + 1)

    def test_deduplication_is_per_participant(self):
        self.add_frame(HASH_A, participant_uuid="participant_1")
        self.add_frame(HASH_A, participant_uuid="participant_2")
        self.assertEqual([kept.participant_uuid for kept in self.kept_frames()], ["participant_1", "participant_2"])

        # Each participant's history is independent
        self.add_frame(HASH_A, participant_uuid="participant_1")
        self.add_frame(HASH_A, participant_uuid="participant_2")
        self.assertEqual(self.on_frame_kept.call_count, 2)

    def test_invalid_base64_is_skipped(self):
        self.capturer.add_frame(frame=b"not base64!", participant_uuid="participant_1", source="screenshare")
        self.on_frame_kept.assert_not_called()

    def test_undecodable_image_is_skipped(self):
        fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"
        self.capturer.add_frame(frame=base64.b64encode(fake_jpeg), participant_uuid="participant_1", source="screenshare")
        self.on_frame_kept.assert_not_called()

    def test_skipped_frames_do_not_enter_the_comparison_window(self):
        self.add_frame(HASH_A)
        near_duplicate = flip_low_bits(HASH_A, MIN_HAMMING_DISTANCE)
        self.add_frame(near_duplicate)
        # A frame that is far from HASH_A but close to the skipped near duplicate must still be judged against HASH_A only
        far_from_a_close_to_duplicate = flip_low_bits(HASH_A, MIN_HAMMING_DISTANCE + 1)
        self.assertEqual(hamming_distance(far_from_a_close_to_duplicate, near_duplicate), 1)
        self.add_frame(far_from_a_close_to_duplicate)
        self.assertEqual([kept.dhash for kept in self.kept_frames()], [HASH_A, far_from_a_close_to_duplicate])
