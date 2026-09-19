// The brain playback clock owns this preview. Each prediction carries the original clip offsets,
// including both pieces when its one-second interval crosses a phone recording boundary.
export function createVideoPreview({ baseUrl, sid, mock = false }) {
  const box = document.getElementById('videoPreview');
  const message = document.getElementById('videoMessage');
  const label = document.getElementById('videoTime');
  const slots = [...box.querySelectorAll('video')].map((video) => ({ video, index: null, playing: false, blocked: false }));
  const base = (baseUrl || '').replace(/\/$/, '');
  let frame = null, nextFrame = null, anchor = 0, period = 1000, current = null;

  function hide(text) {
    message.textContent = text;
    message.hidden = false;
    for (const slot of slots) { slot.video.pause(); slot.video.classList.remove('visible'); }
  }

  function load(index) {
    let slot = slots.find((s) => s.index === index);
    if (slot) return slot;
    slot = slots.find((s) => s !== current);
    slot.video.pause();
    slot.video.classList.remove('visible');
    slot.index = index;
    slot.blocked = false;
    slot.video.muted = true;
    slot.video.src = `${base}/session/${encodeURIComponent(sid)}/video/${index}`;
    slot.video.load();
    return slot;
  }

  function preload() {
    // Keep the second decoder ready for the next chunk; do not replace the visible decoder.
    const candidates = [...(frame?.clips || []), ...(nextFrame?.clips || [])];
    const next = candidates.find((c) => c.index !== current?.index && c.index > (current?.index ?? -1));
    if (next) load(next.index);
  }

  function tick(now = performance.now()) {
    if (!frame || !base || mock) return;
    if (document.hidden || document.body.classList.contains('game')) {
      for (const slot of slots) slot.video.pause();
      return;
    }
    const clips = frame.clips || [];
    const duration = Math.min(1, clips.reduce((n, c) => n + c.duration_s, 0));
    if (!duration) { hide('Video unavailable for this prediction'); return; }

    // Never run beyond this prediction interval, even while the brain waits for more results.
    const elapsed = Math.max(0, (now - anchor) / period);
    const progress = Math.min(elapsed, Math.max(0, duration - 0.035));
    let remaining = progress, clip = clips[clips.length - 1];
    for (const part of clips) {
      clip = part;
      if (remaining < part.duration_s) break;
      remaining -= part.duration_s;
    }
    const slot = load(clip.index), video = slot.video;
    if (current !== slot) {
      if (current) { current.video.pause(); current.video.classList.remove('visible'); }
      current = slot;
      message.hidden = false;
      message.textContent = 'Loading synced video…';
    }
    preload();
    if (video.error) { hide('Video unavailable · brain playback continues'); return; }
    if (video.readyState < 1 || video.seeking) return;

    const target = Math.min(clip.offset_s + remaining, Math.max(0, video.duration - 0.035));
    if (Math.abs(video.currentTime - target) > 0.12) {
      video.currentTime = target;
      return;
    }
    if (video.readyState < 2) return;
    video.classList.add('visible');
    message.hidden = true;
    video.playbackRate = 1000 / period;
    if (elapsed >= duration - 0.035) video.pause();
    else if (video.paused && !slot.playing && !slot.blocked) {
      slot.playing = true;
      const index = slot.index;
      video.play().catch((error) => {
        // Replacing a clip or pausing at the next brain tick can cancel a pending play().
        if (slot.index === index && error.name !== 'AbortError') slot.blocked = true;
      }).finally(() => { slot.playing = false; });
    }
  }

  function follow(prediction, periodMs, upcoming = null) {
    frame = prediction;
    nextFrame = upcoming;
    anchor = performance.now();
    period = Math.max(100, periodMs);
    for (const slot of slots) slot.video.pause();
    if (mock) return;
    label.textContent = `${prediction.second} s`;
    tick(anchor);
  }

  for (const slot of slots) {
    for (const event of ['loadedmetadata', 'loadeddata', 'seeked', 'error']) slot.video.addEventListener(event, () => tick());
  }
  box.addEventListener('click', () => { for (const slot of slots) slot.blocked = false; tick(); });
  hide(mock ? 'No camera video in mock mode' : 'Waiting for the phone');
  const timer = setInterval(tick, 40);
  window.addEventListener('pagehide', () => { clearInterval(timer); for (const slot of slots) slot.video.pause(); }, { once: true });
  return { follow };
}
