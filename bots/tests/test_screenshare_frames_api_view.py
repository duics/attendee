from datetime import timedelta
from unittest.mock import patch
from urllib.parse import urlencode

from django.test import Client, TransactionTestCase
from django.utils import timezone
from rest_framework import status

from accounts.models import Organization
from bots.models import ApiKey, Bot, BotStates, Participant, Project, Recording, RecordingStates, ScreenshareFrame, TranscriptionTypes


def fake_remote_storage_url(file_field):
    return f"https://signed.example.com/{file_field.name}?signature=abc"


class ScreenshareFramesViewTest(TransactionTestCase):
    """Tests for GET /api/v1/bots/{object_id}/screenshare_frames."""

    def setUp(self):
        self.organization_a = Organization.objects.create(name="Organization A")
        self.organization_b = Organization.objects.create(name="Organization B")
        self.project_a = Project.objects.create(name="Project A", organization=self.organization_a)
        self.project_b = Project.objects.create(name="Project B", organization=self.organization_b)
        self.api_key_a, self.api_key_a_plain = ApiKey.create(project=self.project_a, name="API Key A")
        self.api_key_b, self.api_key_b_plain = ApiKey.create(project=self.project_b, name="API Key B")

        self.bot = Bot.objects.create(
            project=self.project_a,
            meeting_url="https://meet.google.com/abc-defg-hij",
            name="Bot A",
            state=BotStates.ENDED,
            settings={"recording_settings": {"record_screenshare_frames": True}},
        )
        self.recording = Recording.objects.create(
            bot=self.bot,
            is_default_recording=True,
            recording_type=self.bot.recording_type(),
            transcription_type=TranscriptionTypes.NON_REALTIME,
            state=RecordingStates.COMPLETE,
            first_buffer_timestamp_ms=1_700_000_000_000,
        )
        self.participant = Participant.objects.create(bot=self.bot, uuid="sharer_uuid", full_name="Sharer")

        self.frame_1 = self.create_frame(timestamp_ms=1_700_000_005_000, participant=self.participant, file_name="screenshare-frames/bot-rec/1700000005000-1.jpg")
        self.frame_2 = self.create_frame(timestamp_ms=1_700_000_009_000, participant=None, file_name="screenshare-frames/bot-rec/1700000009000-2.jpg")
        self.frame_3 = self.create_frame(timestamp_ms=1_700_000_015_000, participant=self.participant, file_name="screenshare-frames/bot-rec/1700000015000-3.jpg")
        # A frame whose image upload has not completed yet
        self.frame_without_file = self.create_frame(timestamp_ms=1_700_000_020_000, participant=self.participant, file_name="")

        self.client = Client()

        remote_storage_url_patcher = patch("bots.models.remote_storage_url", side_effect=fake_remote_storage_url)
        remote_storage_url_patcher.start()
        self.addCleanup(remote_storage_url_patcher.stop)

    def create_frame(self, timestamp_ms, participant, file_name):
        return ScreenshareFrame.objects.create(
            recording=self.recording,
            participant=participant,
            timestamp_ms=timestamp_ms,
            dhash="a5a5a5a5a5a5a5a5",
            width=1920,
            height=1080,
            file=file_name,
        )

    def get(self, url, api_key):
        return self.client.get(url, HTTP_AUTHORIZATION=f"Token {api_key}")

    def test_lists_uploaded_frames_in_capture_order(self):
        response = self.get(f"/api/v1/bots/{self.bot.object_id}/screenshare_frames", self.api_key_a_plain)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        body = response.json()
        self.assertIn("next", body)
        self.assertIn("previous", body)
        results = body["results"]
        self.assertEqual([frame["id"] for frame in results], [self.frame_1.object_id, self.frame_2.object_id, self.frame_3.object_id])

        first = results[0]
        self.assertEqual(
            first,
            {
                "id": self.frame_1.object_id,
                "participant_uuid": "sharer_uuid",
                "timestamp_ms": 1_700_000_005_000,
                "width": 1920,
                "height": 1080,
                "url": "https://signed.example.com/screenshare-frames/bot-rec/1700000005000-1.jpg?signature=abc",
            },
        )

    def test_frame_without_an_identified_participant_has_null_participant_uuid(self):
        response = self.get(f"/api/v1/bots/{self.bot.object_id}/screenshare_frames", self.api_key_a_plain)
        results = response.json()["results"]
        self.assertIsNone(results[1]["participant_uuid"])

    def test_frames_whose_upload_has_not_completed_are_not_listed(self):
        response = self.get(f"/api/v1/bots/{self.bot.object_id}/screenshare_frames", self.api_key_a_plain)
        ids = [frame["id"] for frame in response.json()["results"]]
        self.assertNotIn(self.frame_without_file.object_id, ids)

    def test_other_projects_cannot_see_the_frames(self):
        response = self.get(f"/api/v1/bots/{self.bot.object_id}/screenshare_frames", self.api_key_b_plain)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_unknown_bot_returns_404(self):
        response = self.get("/api/v1/bots/bot_doesnotexist/screenshare_frames", self.api_key_a_plain)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_requires_authentication(self):
        response = self.client.get(f"/api/v1/bots/{self.bot.object_id}/screenshare_frames")
        self.assertIn(response.status_code, [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN])

    def test_after_and_before_filter_on_creation_time(self):
        cutoff = timezone.now()
        ScreenshareFrame.objects.filter(id=self.frame_3.id).update(created_at=cutoff + timedelta(minutes=1))

        response = self.get(f"/api/v1/bots/{self.bot.object_id}/screenshare_frames?{urlencode({'after': cutoff.isoformat()})}", self.api_key_a_plain)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual([frame["id"] for frame in response.json()["results"]], [self.frame_3.object_id])

        response = self.get(f"/api/v1/bots/{self.bot.object_id}/screenshare_frames?{urlencode({'before': cutoff.isoformat()})}", self.api_key_a_plain)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual([frame["id"] for frame in response.json()["results"]], [self.frame_1.object_id, self.frame_2.object_id])

    def test_invalid_after_returns_400(self):
        response = self.get(f"/api/v1/bots/{self.bot.object_id}/screenshare_frames?after=yesterday", self.api_key_a_plain)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_pagination_returns_next_cursor(self):
        for i in range(30):
            self.create_frame(timestamp_ms=1_700_000_100_000 + i, participant=self.participant, file_name=f"screenshare-frames/bot-rec/extra-{i}.jpg")

        response = self.get(f"/api/v1/bots/{self.bot.object_id}/screenshare_frames", self.api_key_a_plain)
        body = response.json()
        self.assertEqual(len(body["results"]), 25)
        self.assertIsNotNone(body["next"])

        next_page = self.client.get(body["next"], HTTP_AUTHORIZATION=f"Token {self.api_key_a_plain}")
        self.assertEqual(next_page.status_code, status.HTTP_200_OK)
        self.assertEqual(len(next_page.json()["results"]), 8)
