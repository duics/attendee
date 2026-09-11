import unittest
from unittest.mock import Mock, patch

from bots.bot_controller.screenshare_frame_uploader import ScreenshareFrameUploader


class FakeStorage:
    def __init__(self, fail_for_filenames=()):
        self.saved = []
        self.fail_for_filenames = set(fail_for_filenames)

    def save(self, name, content):
        if name in self.fail_for_filenames:
            raise IOError(f"could not write {name}")
        self.saved.append((name, content.read()))
        return name


class TestScreenshareFrameUploader(unittest.TestCase):
    def setUp(self):
        self.storage = FakeStorage()
        storages_patcher = patch("bots.bot_controller.screenshare_frame_uploader.storages", {"recordings": self.storage})
        storages_patcher.start()
        self.addCleanup(storages_patcher.stop)

        self.on_success = Mock()
        self.on_error = Mock()
        self.uploader = ScreenshareFrameUploader(on_success=self.on_success, on_error=self.on_error, max_workers=1)
        self.addCleanup(self.uploader.shutdown)

    def test_uses_the_recording_storage(self):
        self.assertIs(self.uploader.storage, self.storage)

    def test_successful_upload_reports_stored_name(self):
        self.uploader.upload(screenshare_frame_id=7, filename="screenshare-frames/bot-rec/1000-7.jpg", data=b"jpeg bytes")
        self.uploader.wait_for_uploads(timeout=5)

        self.assertEqual(self.storage.saved, [("screenshare-frames/bot-rec/1000-7.jpg", b"jpeg bytes")])
        self.on_success.assert_called_once_with(7, "screenshare-frames/bot-rec/1000-7.jpg")
        self.on_error.assert_not_called()

    def test_failed_upload_reports_error(self):
        self.storage.fail_for_filenames.add("screenshare-frames/bot-rec/1000-8.jpg")
        self.uploader.upload(screenshare_frame_id=8, filename="screenshare-frames/bot-rec/1000-8.jpg", data=b"jpeg bytes")
        self.uploader.wait_for_uploads(timeout=5)

        self.on_success.assert_not_called()
        self.on_error.assert_called_once()
        self.assertEqual(self.on_error.call_args.kwargs["screenshare_frame_id"], 8)
        self.assertIsInstance(self.on_error.call_args.kwargs["exception"], IOError)

    def test_results_are_delivered_once(self):
        self.uploader.upload(screenshare_frame_id=9, filename="a.jpg", data=b"a")
        self.uploader.wait_for_uploads(timeout=5)
        self.uploader.process_uploads()
        self.uploader.process_uploads()

        self.on_success.assert_called_once_with(9, "a.jpg")

    def test_upload_after_shutdown_reports_error(self):
        self.uploader.shutdown()
        self.uploader.upload(screenshare_frame_id=10, filename="b.jpg", data=b"b")

        self.on_success.assert_not_called()
        self.on_error.assert_called_once()
        self.assertEqual(self.on_error.call_args.kwargs["screenshare_frame_id"], 10)

    def test_callback_exceptions_do_not_stop_processing(self):
        self.on_success.side_effect = RuntimeError("boom")
        self.uploader.upload(screenshare_frame_id=11, filename="c.jpg", data=b"c")
        self.uploader.upload(screenshare_frame_id=12, filename="d.jpg", data=b"d")
        self.uploader.wait_for_uploads(timeout=5)

        self.assertEqual(self.on_success.call_count, 2)
        self.assertEqual(len(self.storage.saved), 2)
