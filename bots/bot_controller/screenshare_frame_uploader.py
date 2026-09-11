import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

from django.core.files.base import ContentFile
from django.core.files.storage import storages


class ScreenshareFrameUploader:
    """
    Simple in-process async uploader for screenshare frame images, mirroring AudioChunkUploader.

    - Uploads are queued via upload() and processed via process_uploads() from the main thread
    """

    def __init__(
        self,
        on_success: Callable[[int, str], None],
        on_error: Optional[Callable[[int, Exception], None]] = None,
        max_workers: int = 2,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            on_success: Called with (screenshare_frame_id, stored_name) for each successful upload
            on_error: Called with (screenshare_frame_id, exception) for each failed upload
            max_workers: Maximum number of concurrent upload threads
            logger: Optional logger instance
        """
        self._on_success = on_success
        self._on_error = on_error
        self.storage = storages["recordings"]
        self.log = logger or logging.getLogger(__name__)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="screenshare-frame-uploader")
        self._lock = threading.Lock()
        self._pending_uploads: Dict[int, Dict] = {}  # screenshare_frame_id -> upload info

    def upload(self, screenshare_frame_id: int, filename: str, data: bytes):
        try:
            fut = self._pool.submit(self._upload_one, filename, data)
        except Exception as e:
            # executor is shut down (or broken)
            self.log.warning("Upload rejected (executor shut down) screenshare_frame_id=%s", screenshare_frame_id)
            self._call_on_error(screenshare_frame_id, e)
            return

        with self._lock:
            self._pending_uploads[screenshare_frame_id] = {"future": fut, "filename": filename}
            inflight = len(self._pending_uploads)

        self.log.info("ScreenshareFrameUploader queued screenshare_frame_id=%s, inflight=%s", screenshare_frame_id, inflight)

    def process_uploads(self):
        """
        Delivers the results of completed uploads. Call this from the main thread at regular intervals.
        """
        completed = []

        with self._lock:
            for screenshare_frame_id, upload_info in list(self._pending_uploads.items()):
                if upload_info["future"].done():
                    completed.append((screenshare_frame_id, upload_info))
                    del self._pending_uploads[screenshare_frame_id]

        for screenshare_frame_id, upload_info in completed:
            try:
                stored_name = upload_info["future"].result()
            except Exception as e:
                self.log.exception("Upload failed for screenshare_frame_id=%s, filename=%s", screenshare_frame_id, upload_info["filename"])
                self._call_on_error(screenshare_frame_id, e)
                continue

            try:
                self._on_success(screenshare_frame_id, stored_name)
            except Exception:
                self.log.exception("on_success callback failed for screenshare_frame_id=%s", screenshare_frame_id)

    def _call_on_error(self, screenshare_frame_id: int, exception: Exception):
        if not self._on_error:
            return
        try:
            self._on_error(screenshare_frame_id=screenshare_frame_id, exception=exception)
        except Exception:
            self.log.exception("on_error callback failed for screenshare_frame_id=%s", screenshare_frame_id)

    def _upload_one(self, filename: str, data: bytes) -> str:
        # storage.save may alter the name (avoid collisions), so use return value
        return self.storage.save(filename, ContentFile(data))

    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)

    def wait_for_uploads(self, timeout: float = 5.0):
        """
        Waits for all pending uploads to complete, delivering their results as they finish.
        Uploads still pending when the timeout elapses are reported through on_error.
        """
        start_time = time.time()
        while True:
            self.process_uploads()

            with self._lock:
                pending_count = len(self._pending_uploads)
                pending_ids = list(self._pending_uploads.keys())

            if pending_count == 0:
                self.log.info("wait_for_uploads: all screenshare frame uploads completed")
                return

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                self.log.warning("wait_for_uploads: timeout after %.1fs with %d screenshare frame uploads still pending", elapsed, pending_count)
                for screenshare_frame_id in pending_ids:
                    self._call_on_error(screenshare_frame_id, Exception("In wait_for_uploads, upload timed out"))
                return

            time.sleep(0.1)
