import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from bots.per_participant_realtime_video_configuration import (
    PerParticipantRealtimeVideoConfiguration,
    PerParticipantRealtimeVideoSourceConfiguration,
)
from bots.per_participant_realtime_video_frame_change_detector import (
    FINGERPRINT_HEIGHT,
    FINGERPRINT_WIDTH,
    FRAME_CHANGE_THRESHOLD,
    PerParticipantRealtimeVideoFrameChangeDetector,
    compute_frame_fingerprint,
    frame_fingerprint_changed,
)
from bots.zoom_bot_adapter.realtime_per_participant_video_frame_generator import (
    RealtimePerParticipantVideoFrameGenerator,
    _PerParticipantVideoFrameSubscription,
)

WIDTH = 640
HEIGHT = 360


def make_slide(text_block_x: int) -> np.ndarray:
    """A white 'slide' with a dark block at a given horizontal position."""
    bgr = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    bgr[100:260, text_block_x : text_block_x + 200, :] = 40
    return bgr


def add_noise(bgr: np.ndarray, amplitude: int, seed: int = 0) -> np.ndarray:
    """Simulate encoder noise: every pixel moves by at most `amplitude` levels."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-amplitude, amplitude + 1, size=bgr.shape, dtype=np.int16)
    return np.clip(bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)


class I420Frame:
    """Mimics a Zoom raw video frame object for a given BGR image."""

    def __init__(self, bgr: np.ndarray):
        height, width, _ = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).tobytes()
        y_size = width * height
        uv_size = (width // 2) * (height // 2)
        self._width = width
        self._height = height
        self._y = yuv[:y_size]
        self._u = yuv[y_size : y_size + uv_size]
        self._v = yuv[y_size + uv_size : y_size + 2 * uv_size]

    def GetStreamWidth(self):
        return self._width

    def GetStreamHeight(self):
        return self._height

    def GetYBuffer(self):
        return self._y

    def GetUBuffer(self):
        return self._u

    def GetVBuffer(self):
        return self._v


class TestFrameFingerprintFunctions(unittest.TestCase):
    def test_fingerprint_is_a_small_float_thumbnail(self):
        fingerprint = compute_frame_fingerprint(make_slide(100))

        self.assertEqual(fingerprint.shape, (FINGERPRINT_HEIGHT, FINGERPRINT_WIDTH))
        self.assertEqual(fingerprint.dtype, np.float32)

    def test_first_frame_is_always_a_change(self):
        self.assertTrue(frame_fingerprint_changed(None, compute_frame_fingerprint(make_slide(100))))

    def test_identical_frames_are_not_a_change(self):
        fingerprint = compute_frame_fingerprint(make_slide(100))

        self.assertFalse(frame_fingerprint_changed(fingerprint, fingerprint.copy()))

    def test_encoder_noise_is_not_a_change(self):
        clean = compute_frame_fingerprint(make_slide(100))
        noisy = compute_frame_fingerprint(add_noise(make_slide(100), amplitude=3))

        self.assertFalse(frame_fingerprint_changed(clean, noisy))

    def test_slide_change_is_a_change(self):
        first = compute_frame_fingerprint(make_slide(100))
        second = compute_frame_fingerprint(make_slide(300))

        self.assertTrue(frame_fingerprint_changed(first, second))

    def test_threshold_is_inclusive_and_configurable(self):
        base = np.zeros((FINGERPRINT_HEIGHT, FINGERPRINT_WIDTH), dtype=np.float32)
        shifted = base + FRAME_CHANGE_THRESHOLD

        self.assertTrue(frame_fingerprint_changed(base, shifted))
        self.assertFalse(frame_fingerprint_changed(base, shifted, threshold=FRAME_CHANGE_THRESHOLD + 1))

    def test_shape_mismatch_is_a_change(self):
        current = compute_frame_fingerprint(make_slide(100))
        previous = np.zeros((4, 4), dtype=np.float32)

        self.assertTrue(frame_fingerprint_changed(previous, current))


class TestPerParticipantRealtimeVideoFrameChangeDetector(unittest.TestCase):
    def test_compares_against_last_sent_frame_only(self):
        detector = PerParticipantRealtimeVideoFrameChangeDetector()
        slide_a = make_slide(100)
        slide_b = make_slide(300)

        # Nothing sent yet: every frame counts as changed until one is marked sent
        self.assertTrue(detector.frame_changed(slide_a))
        self.assertTrue(detector.frame_changed(slide_a))

        detector.mark_sent()

        self.assertFalse(detector.frame_changed(slide_a))
        self.assertFalse(detector.frame_changed(add_noise(slide_a, amplitude=3)))
        self.assertTrue(detector.frame_changed(slide_b))

        # A changed frame that was not sent does not move the baseline
        self.assertFalse(detector.frame_changed(slide_a))

        self.assertTrue(detector.frame_changed(slide_b))
        detector.mark_sent()
        self.assertFalse(detector.frame_changed(slide_b))
        self.assertTrue(detector.frame_changed(slide_a))


@patch("bots.zoom_bot_adapter.realtime_per_participant_video_frame_generator.zoom")
class TestZoomSubscriptionFrameDelivery(unittest.TestCase):
    def setUp(self):
        self.frame_callback = Mock()
        self.recording_is_paused = Mock(return_value=False)
        self.generator = RealtimePerParticipantVideoFrameGenerator(
            frame_callback=self.frame_callback,
            get_participants_ctrl_callback=Mock(),
            get_meeting_sharing_controller_callback=Mock(),
            get_recording_is_paused_callback=self.recording_is_paused,
            per_participant_realtime_video_configuration=PerParticipantRealtimeVideoConfiguration(),
        )

    def make_subscription(self, frame_delivery: str, share_source_id=None):
        subscription = _PerParticipantVideoFrameSubscription(
            owner=self.generator,
            participant_id=2,
            share_source_id=share_source_id,
            source_configuration=PerParticipantRealtimeVideoSourceConfiguration(resolution="360p", frame_delivery=frame_delivery),
        )
        # Disable the sampling clock so every frame in the test is a sample
        subscription.min_frame_interval_ns = 0
        return subscription

    def test_continuous_sends_every_sampled_frame(self, mock_zoom):
        subscription = self.make_subscription("continuous")

        self.assertIsNone(subscription._change_detector)

        subscription._on_raw_video_frame_received(I420Frame(make_slide(100)))
        subscription._on_raw_video_frame_received(I420Frame(make_slide(100)))

        self.assertEqual(self.frame_callback.call_count, 2)

    def test_on_change_skips_unchanged_frames(self, mock_zoom):
        subscription = self.make_subscription("on_change", share_source_id=42)

        subscription._on_raw_video_frame_received(I420Frame(make_slide(100)))
        subscription._on_raw_video_frame_received(I420Frame(make_slide(100)))
        subscription._on_raw_video_frame_received(I420Frame(add_noise(make_slide(100), amplitude=3)))

        self.assertEqual(self.frame_callback.call_count, 1)

        subscription._on_raw_video_frame_received(I420Frame(make_slide(300)))

        self.assertEqual(self.frame_callback.call_count, 2)
        _, participant_id, source = self.frame_callback.call_args.args
        self.assertEqual(participant_id, 2)
        self.assertEqual(source, "screenshare")

    def test_on_change_resends_a_frame_that_was_not_delivered(self, mock_zoom):
        subscription = self.make_subscription("on_change")

        self.recording_is_paused.return_value = True
        subscription._on_raw_video_frame_received(I420Frame(make_slide(100)))
        self.assertEqual(self.frame_callback.call_count, 0)

        # Once frames can be delivered again, the same picture is still owed to the consumer
        self.recording_is_paused.return_value = False
        subscription._on_raw_video_frame_received(I420Frame(make_slide(100)))
        self.assertEqual(self.frame_callback.call_count, 1)

        subscription._on_raw_video_frame_received(I420Frame(make_slide(100)))
        self.assertEqual(self.frame_callback.call_count, 1)
