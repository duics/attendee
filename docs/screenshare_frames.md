# Screenshare Frames

Attendee can keep the distinct screens that were shared during a meeting as JPEG images, stored with the recording and listed through the API. This gives you the slides, documents and demos that were shown without processing the recording yourself. The feature is off by default.

## Setup

Set `recording_settings.record_screenshare_frames` to `true` when creating a bot:

```json
{
  "meeting_url": "https://us06web.zoom.us/j/12345678",
  "bot_name": "Screenshare Bot",
  "recording_settings": {
    "record_screenshare_frames": true
  }
}
```

The recording format must be `mp4` or `mp3`; requests with `"format": "none"` are rejected. RTMP streaming bots do not capture screenshare frames. When combined with [realtime video](realtime_video.md), the websocket receives screenshare frames at 1080p regardless of `screenshare_resolution`, because the same frames feed both.

## How frames are selected

The bot receives the screenshare video at 1080p and one frame per second. Each frame is hashed with a 64-bit difference hash (dHash) and kept only if it differs by more than 8 bits from each of the last 5 frames kept for that participant. A slide that stays on screen produces one frame; a slide change, scroll or window switch produces a new one; cursor movement does not. Frames that arrive before the recording starts or while it is paused are not captured.

Frames are attributed to the participant who was sharing. If the sharer cannot be identified, the frame is kept with a `null` participant.

## Listing frames

Send a GET request to `/api/v1/bots/{bot_id}/screenshare_frames`. Frames are returned in capture order with cursor pagination. The optional `after` and `before` query parameters accept ISO 8601 timestamps and filter on when the frame was created, which is useful for polling during the meeting.

```json
{
  "next": null,
  "previous": null,
  "results": [
    {
      "id": "frame_xxxxxxxxxxxxxxxx",
      "participant_uuid": "participant_abc123",
      "timestamp_ms": 1733114771000,
      "width": 1920,
      "height": 1080,
      "url": "https://.../screenshare-frames/bot_xxx-rec_xxx/1733114771000-1.jpg?..."
    }
  ]
}
```

`timestamp_ms` is milliseconds since the Unix epoch, on the same clock as transcript utterances and participant events; subtract the recording's `start_timestamp_ms` for the offset into the recording. `url` expires after 30 minutes, so request the list again when you need a fresh link. Frames appear in the list once their image has been uploaded.

## Storage and deletion

Images are stored in the recording storage bucket under `screenshare-frames/{bot_id}-{recording_id}/`. Deleting a bot's data deletes the frames and their images together with the recording and transcript.
