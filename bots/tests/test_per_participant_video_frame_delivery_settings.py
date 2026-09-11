from django.test import SimpleTestCase
from rest_framework.exceptions import ValidationError

from bots.models import Bot
from bots.serializers import CreateBotSerializer

VIDEO_URL = "wss://example.com/per-participant-video"


class TestPerParticipantVideoFrameDeliverySerializerValidation(SimpleTestCase):
    def setUp(self):
        self.serializer = CreateBotSerializer()

    def test_frame_delivery_can_be_omitted(self):
        value = {"per_participant_video": {"url": VIDEO_URL}}

        validated = self.serializer.validate_websocket_settings(value)

        self.assertEqual(validated, {"per_participant_video": {"url": VIDEO_URL}})

    def test_on_change_is_accepted_per_source(self):
        value = {
            "per_participant_video": {
                "url": VIDEO_URL,
                "webcam_resolution": "360p",
                "screenshare_resolution": "720p",
                "webcam_frame_delivery": "continuous",
                "screenshare_frame_delivery": "on_change",
            }
        }

        validated = self.serializer.validate_websocket_settings(value)

        self.assertEqual(validated["per_participant_video"]["webcam_frame_delivery"], "continuous")
        self.assertEqual(validated["per_participant_video"]["screenshare_frame_delivery"], "on_change")

    def test_invalid_webcam_frame_delivery_is_rejected(self):
        value = {"per_participant_video": {"url": VIDEO_URL, "webcam_frame_delivery": "sometimes"}}

        with self.assertRaises(ValidationError):
            self.serializer.validate_websocket_settings(value)

    def test_invalid_screenshare_frame_delivery_is_rejected(self):
        value = {"per_participant_video": {"url": VIDEO_URL, "screenshare_frame_delivery": True}}

        with self.assertRaises(ValidationError):
            self.serializer.validate_websocket_settings(value)

    def test_unknown_per_participant_video_keys_are_still_rejected(self):
        value = {"per_participant_video": {"url": VIDEO_URL, "frame_delivery": "on_change"}}

        with self.assertRaises(ValidationError):
            self.serializer.validate_websocket_settings(value)


class TestBotPerParticipantVideoFrameDeliveryAccessors(SimpleTestCase):
    def test_defaults_to_continuous_when_unset(self):
        bot = Bot(settings={"websocket_settings": {"per_participant_video": {"url": VIDEO_URL}}})

        self.assertEqual(bot.websocket_per_participant_video_webcam_frame_delivery(), "continuous")
        self.assertEqual(bot.websocket_per_participant_video_screenshare_frame_delivery(), "continuous")

    def test_defaults_to_continuous_without_websocket_settings(self):
        bot = Bot(settings={})

        self.assertEqual(bot.websocket_per_participant_video_webcam_frame_delivery(), "continuous")
        self.assertEqual(bot.websocket_per_participant_video_screenshare_frame_delivery(), "continuous")

    def test_returns_configured_values_per_source(self):
        bot = Bot(
            settings={
                "websocket_settings": {
                    "per_participant_video": {
                        "url": VIDEO_URL,
                        "webcam_frame_delivery": "on_change",
                        "screenshare_frame_delivery": "continuous",
                    }
                }
            }
        )

        self.assertEqual(bot.websocket_per_participant_video_webcam_frame_delivery(), "on_change")
        self.assertEqual(bot.websocket_per_participant_video_screenshare_frame_delivery(), "continuous")
