from unittest.mock import Mock

from django.test import TestCase
from rest_framework import serializers as drf_serializers

from accounts.models import Organization
from bots.bot_controller.bot_controller import BotController
from bots.bot_controller.pipeline_configuration import PipelineConfiguration
from bots.bot_controller.screenshare_frame_capturer import KeptScreenshareFrame
from bots.models import Bot, BotStates, Participant, Project, Recording, RecordingStates, ScreenshareFrame, TranscriptionTypes
from bots.serializers import BOT_RECORDING_SETTINGS_DEFAULT_VALUES, CreateBotSerializer


class TestRecordScreenshareFramesSetting(TestCase):
    def setUp(self):
        organization = Organization.objects.create(name="Test Organization")
        self.project = Project.objects.create(name="Test Project", organization=organization)

    def create_bot(self, settings):
        return Bot.objects.create(project=self.project, name="Test Bot", meeting_url="https://meet.google.com/abc-defg-hij", settings=settings)

    def test_defaults_to_false(self):
        self.assertFalse(BOT_RECORDING_SETTINGS_DEFAULT_VALUES["record_screenshare_frames"])
        self.assertFalse(self.create_bot({}).record_screenshare_frames())
        self.assertFalse(self.create_bot({"recording_settings": None}).record_screenshare_frames())
        self.assertFalse(self.create_bot({"recording_settings": {"format": "mp4"}}).record_screenshare_frames())

    def test_reads_the_setting(self):
        self.assertTrue(self.create_bot({"recording_settings": {"record_screenshare_frames": True}}).record_screenshare_frames())

    def test_serializer_fills_in_the_default(self):
        value = CreateBotSerializer().validate_recording_settings({"format": "mp4"})
        self.assertFalse(value["record_screenshare_frames"])

    def test_serializer_accepts_the_setting_with_a_recording(self):
        for recording_format in ("mp4", "mp3"):
            value = CreateBotSerializer().validate_recording_settings({"format": recording_format, "record_screenshare_frames": True})
            self.assertTrue(value["record_screenshare_frames"])

    def test_serializer_rejects_the_setting_without_a_recording(self):
        with self.assertRaises(drf_serializers.ValidationError) as context:
            CreateBotSerializer().validate_recording_settings({"format": "none", "record_screenshare_frames": True})
        self.assertIn("record_screenshare_frames", context.exception.detail)

    def test_serializer_rejects_non_boolean_values(self):
        with self.assertRaises(drf_serializers.ValidationError):
            CreateBotSerializer().validate_recording_settings({"record_screenshare_frames": "yes"})


class TestPipelineConfigurationCaptureScreenshareFrames(TestCase):
    def test_flag_defaults_to_false(self):
        self.assertFalse(PipelineConfiguration.recorder_bot().capture_screenshare_frames)
        self.assertFalse(PipelineConfiguration.audio_recorder_bot().capture_screenshare_frames)
        self.assertFalse(PipelineConfiguration.pure_transcription_bot().capture_screenshare_frames)
        self.assertFalse(PipelineConfiguration.rtmp_streaming_bot().capture_screenshare_frames)

    def test_flag_is_a_valid_optional_add_on(self):
        self.assertTrue(PipelineConfiguration.recorder_bot(capture_screenshare_frames=True).capture_screenshare_frames)
        self.assertTrue(PipelineConfiguration.audio_recorder_bot(capture_screenshare_frames=True).capture_screenshare_frames)
        combined = PipelineConfiguration.recorder_bot(websocket_stream_per_participant_video=True, capture_screenshare_frames=True)
        self.assertTrue(combined.capture_screenshare_frames)
        self.assertTrue(combined.websocket_stream_per_participant_video)


class TestBotControllerScreenshareFrameWiring(TestCase):
    def setUp(self):
        organization = Organization.objects.create(name="Test Organization")
        self.project = Project.objects.create(name="Test Project", organization=organization)

    def build_controller(self, settings):
        bot = Bot.objects.create(project=self.project, name="Test Bot", meeting_url="https://meet.google.com/abc-defg-hij", state=BotStates.JOINED_RECORDING, settings=settings)
        controller = BotController.__new__(BotController)
        controller.bot_in_db = bot
        controller.pipeline_configuration = controller.get_pipeline_configuration()
        controller.per_participant_realtime_video_configuration = controller.get_per_participant_realtime_video_configuration()
        controller.screenshare_frame_capturer = None
        controller.screenshare_frame_uploader = None
        controller.websocket_client_manager = None
        return controller

    def test_setting_off_leaves_the_video_path_unchanged(self):
        controller = self.build_controller({"recording_settings": {"format": "mp3"}})

        self.assertFalse(controller.pipeline_configuration.capture_screenshare_frames)
        self.assertIsNone(controller.get_per_participant_video_frame_callback())
        self.assertTrue(controller.disable_incoming_video_for_web_bots())
        self.assertEqual(controller.per_participant_realtime_video_configuration.webcam_configuration.resolution, "360p")
        self.assertEqual(controller.per_participant_realtime_video_configuration.screenshare_configuration.resolution, "360p")

    def test_setting_on_wires_screenshare_only_video_at_1080p(self):
        controller = self.build_controller({"recording_settings": {"format": "mp3", "record_screenshare_frames": True}})

        self.assertTrue(controller.pipeline_configuration.capture_screenshare_frames)
        self.assertFalse(controller.pipeline_configuration.websocket_stream_per_participant_video)
        self.assertIsNotNone(controller.get_per_participant_video_frame_callback())
        self.assertFalse(controller.disable_incoming_video_for_web_bots())
        self.assertEqual(controller.per_participant_realtime_video_configuration.webcam_configuration.resolution, "none")
        self.assertEqual(controller.per_participant_realtime_video_configuration.screenshare_configuration.resolution, "1080p")

    def test_setting_on_with_websocket_video_keeps_webcam_and_raises_screenshare_to_1080p(self):
        controller = self.build_controller(
            {
                "recording_settings": {"format": "mp4", "record_screenshare_frames": True},
                "websocket_settings": {"per_participant_video": {"url": "wss://example.com/video", "webcam_resolution": "720p", "screenshare_resolution": "360p"}},
            }
        )

        self.assertTrue(controller.pipeline_configuration.capture_screenshare_frames)
        self.assertTrue(controller.pipeline_configuration.websocket_stream_per_participant_video)
        self.assertEqual(controller.per_participant_realtime_video_configuration.webcam_configuration.resolution, "720p")
        self.assertEqual(controller.per_participant_realtime_video_configuration.screenshare_configuration.resolution, "1080p")

    def test_websocket_video_without_the_setting_is_unchanged(self):
        controller = self.build_controller({"websocket_settings": {"per_participant_video": {"url": "wss://example.com/video", "webcam_resolution": "720p", "screenshare_resolution": "360p"}}})

        self.assertFalse(controller.pipeline_configuration.capture_screenshare_frames)
        self.assertIsNotNone(controller.get_per_participant_video_frame_callback())
        self.assertEqual(controller.per_participant_realtime_video_configuration.webcam_configuration.resolution, "720p")
        self.assertEqual(controller.per_participant_realtime_video_configuration.screenshare_configuration.resolution, "360p")

    def test_rtmp_bots_do_not_capture(self):
        controller = self.build_controller({"recording_settings": {"record_screenshare_frames": True}, "rtmp_settings": {"destination_url": "rtmp://example.com/live", "stream_key": "key"}})
        self.assertFalse(controller.pipeline_configuration.capture_screenshare_frames)

    def test_frame_callback_feeds_the_capturer_and_the_websocket(self):
        controller = self.build_controller({"recording_settings": {"record_screenshare_frames": True}})
        controller.screenshare_frame_capturer = Mock()
        controller.websocket_client_manager = Mock()

        controller.add_per_participant_video_frame_callback(b"ZmFrZQ==", "participant_1", "screenshare")

        controller.screenshare_frame_capturer.add_frame.assert_called_once_with(frame=b"ZmFrZQ==", participant_uuid="participant_1", source="screenshare")
        controller.websocket_client_manager.send_per_participant_video.assert_called_once()

    def test_frame_callback_without_capturer_only_feeds_the_websocket(self):
        controller = self.build_controller({"websocket_settings": {"per_participant_video": {"url": "wss://example.com/video"}}})
        controller.websocket_client_manager = Mock()

        controller.add_per_participant_video_frame_callback(b"ZmFrZQ==", "participant_1", "webcam")

        controller.websocket_client_manager.send_per_participant_video.assert_called_once()


class TestBotControllerSaveScreenshareFrame(TestCase):
    def setUp(self):
        organization = Organization.objects.create(name="Test Organization")
        self.project = Project.objects.create(name="Test Project", organization=organization)
        self.bot = Bot.objects.create(
            project=self.project,
            name="Test Bot",
            meeting_url="https://meet.google.com/abc-defg-hij",
            state=BotStates.JOINED_RECORDING,
            settings={"recording_settings": {"record_screenshare_frames": True}},
        )
        self.recording = Recording.objects.create(
            bot=self.bot,
            is_default_recording=True,
            recording_type=self.bot.recording_type(),
            transcription_type=TranscriptionTypes.NON_REALTIME,
            state=RecordingStates.IN_PROGRESS,
        )

        self.controller = BotController.__new__(BotController)
        self.controller.bot_in_db = self.bot
        self.controller.adapter = Mock()
        self.controller.adapter.get_participant.return_value = {
            "participant_uuid": "sharer_uuid",
            "participant_full_name": "Sharer",
            "participant_user_uuid": None,
            "participant_is_the_bot": False,
            "participant_is_host": False,
        }
        self.controller.screenshare_frame_uploader = Mock()

        self.kept_frame = KeptScreenshareFrame(jpeg_bytes=b"jpeg bytes", participant_uuid="sharer_uuid", timestamp_ms=1_700_000_005_000, dhash=0xA5A5A5A5A5A5A5A5, width=1920, height=1080)

    def test_creates_the_frame_and_queues_the_upload(self):
        self.controller.save_screenshare_frame(self.kept_frame)

        frame = ScreenshareFrame.objects.get(recording=self.recording)
        self.assertEqual(frame.participant.uuid, "sharer_uuid")
        self.assertEqual(frame.participant.full_name, "Sharer")
        self.assertEqual(frame.timestamp_ms, 1_700_000_005_000)
        self.assertEqual(frame.dhash, "a5a5a5a5a5a5a5a5")
        self.assertEqual(frame.width, 1920)
        self.assertEqual(frame.height, 1080)
        self.assertFalse(frame.file)
        self.assertTrue(frame.object_id.startswith("frame_"))

        self.controller.screenshare_frame_uploader.upload.assert_called_once_with(
            screenshare_frame_id=frame.id,
            filename=f"screenshare-frames/{self.bot.object_id}-{self.recording.object_id}/1700000005000-{frame.id}.jpg",
            data=b"jpeg bytes",
        )

    def test_reuses_an_existing_participant(self):
        participant = Participant.objects.create(bot=self.bot, uuid="sharer_uuid", full_name="Existing Name")

        self.controller.save_screenshare_frame(self.kept_frame)

        frame = ScreenshareFrame.objects.get(recording=self.recording)
        self.assertEqual(frame.participant, participant)
        self.assertEqual(Participant.objects.filter(bot=self.bot).count(), 1)

    def test_keeps_the_frame_when_the_participant_is_unknown(self):
        self.controller.adapter.get_participant.return_value = None

        self.controller.save_screenshare_frame(self.kept_frame)

        frame = ScreenshareFrame.objects.get(recording=self.recording)
        self.assertIsNone(frame.participant)
        self.controller.screenshare_frame_uploader.upload.assert_called_once()

    def test_skips_the_frame_when_no_recording_is_in_progress(self):
        self.recording.state = RecordingStates.COMPLETE
        self.recording.save()

        self.controller.save_screenshare_frame(self.kept_frame)

        self.assertEqual(ScreenshareFrame.objects.count(), 0)
        self.controller.screenshare_frame_uploader.upload.assert_not_called()

    def test_upload_success_sets_the_file(self):
        self.controller.save_screenshare_frame(self.kept_frame)
        frame = ScreenshareFrame.objects.get(recording=self.recording)

        self.controller.on_screenshare_frame_upload_success(frame.id, "screenshare-frames/stored.jpg")

        frame.refresh_from_db()
        self.assertEqual(frame.file.name, "screenshare-frames/stored.jpg")

    def test_upload_error_discards_the_frame(self):
        self.controller.save_screenshare_frame(self.kept_frame)
        frame = ScreenshareFrame.objects.get(recording=self.recording)

        self.controller.on_screenshare_frame_upload_error(screenshare_frame_id=frame.id, exception=Exception("upload failed"))

        self.assertEqual(ScreenshareFrame.objects.count(), 0)
