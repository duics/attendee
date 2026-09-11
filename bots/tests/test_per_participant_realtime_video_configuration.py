import unittest

from bots.per_participant_realtime_video_configuration import (
    PerParticipantRealtimeVideoConfiguration,
    PerParticipantRealtimeVideoSourceConfiguration,
)


class TestPerParticipantRealtimeVideoSourceConfiguration(unittest.TestCase):
    def test_frame_delivery_defaults_to_continuous(self):
        config = PerParticipantRealtimeVideoSourceConfiguration()

        self.assertEqual(config.frame_delivery, "continuous")
        self.assertFalse(config.send_only_on_change)

    def test_on_change_frame_delivery(self):
        config = PerParticipantRealtimeVideoSourceConfiguration(resolution="720p", frame_delivery="on_change")

        self.assertTrue(config.send_only_on_change)
        # Resolution-derived parameters are unaffected by the delivery mode
        self.assertEqual(config.width, 1280)
        self.assertEqual(config.height, 720)
        self.assertEqual(config.framerate, 1.0)
        self.assertEqual(config.jpeg_quality, 60)

    def test_on_change_with_disabled_source(self):
        config = PerParticipantRealtimeVideoSourceConfiguration(resolution="none", frame_delivery="on_change")

        self.assertFalse(config.enabled)
        self.assertTrue(config.send_only_on_change)

    def test_invalid_frame_delivery_raises(self):
        with self.assertRaises(ValueError) as ctx:
            PerParticipantRealtimeVideoSourceConfiguration(frame_delivery="sometimes")

        self.assertIn("Invalid frame_delivery: sometimes", str(ctx.exception))
        self.assertIn("continuous, on_change", str(ctx.exception))

    def test_invalid_resolution_still_raises(self):
        with self.assertRaises(ValueError):
            PerParticipantRealtimeVideoSourceConfiguration(resolution="4k")

    def test_to_dict_includes_frame_delivery(self):
        config = PerParticipantRealtimeVideoSourceConfiguration(resolution="360p", frame_delivery="on_change")

        self.assertEqual(
            config.to_dict(),
            {
                "resolution": "360p",
                "frame_delivery": "on_change",
                "width": 640,
                "height": 360,
                "framerate": 2.0,
                "jpeg_quality": 70,
                "enabled": True,
            },
        )


class TestPerParticipantRealtimeVideoConfiguration(unittest.TestCase):
    def test_defaults_are_continuous_360p(self):
        config = PerParticipantRealtimeVideoConfiguration()

        self.assertEqual(config.webcam_configuration.frame_delivery, "continuous")
        self.assertEqual(config.screenshare_configuration.frame_delivery, "continuous")
        self.assertEqual(config.webcam_configuration.resolution, "360p")
        self.assertEqual(config.screenshare_configuration.resolution, "360p")

    def test_to_dict_nests_frame_delivery_per_source(self):
        config = PerParticipantRealtimeVideoConfiguration(
            webcam_configuration=PerParticipantRealtimeVideoSourceConfiguration(resolution="360p"),
            screenshare_configuration=PerParticipantRealtimeVideoSourceConfiguration(resolution="1080p", frame_delivery="on_change"),
        )

        as_dict = config.to_dict()

        self.assertEqual(as_dict["webcam_configuration"]["frame_delivery"], "continuous")
        self.assertEqual(as_dict["screenshare_configuration"]["frame_delivery"], "on_change")
        self.assertEqual(as_dict["screenshare_configuration"]["width"], 1920)
