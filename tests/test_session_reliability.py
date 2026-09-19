"""Real ffmpeg regression for the recorder overflow, plus retry-safe session finishing."""
import concurrent.futures
import json
import shutil
import struct
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

from app import server
from modal_app import tribe_service as service


def media_json(path, entries):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', entries, '-show_data_hash', 'sha256',
        '-of', 'json', str(path)], text=True))


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg required')
class ChunkTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.clean = cls.root / 'clean.mp4'
        subprocess.run([
            'ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc=size=160x120:rate=30',
            '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-t', '3',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
            '-video_track_timescale', '600', '-movflags', '+frag_every_frame+empty_moov+default_base_moof',
            str(cls.clean)], check=True, capture_output=True)
        # Reproduce the observed phone bug: the last video fragment stores -5 ticks as uint32.
        data = bytearray(cls.clean.read_bytes())
        video_headers = []

        def visit(lo, hi):
            while lo + 8 <= hi:
                size, kind = struct.unpack_from('>I4s', data, lo)
                if size < 8:
                    break
                if kind in {b'moof', b'traf'}:
                    visit(lo + 8, lo + size)
                if kind == b'tfhd' and struct.unpack_from('>I', data, lo + 12)[0] == 1:
                    video_headers.append(lo)
                lo += size

        visit(0, len(data))
        last = video_headers[-1]
        assert struct.unpack_from('>I', data, last + 8)[0] & 0x000008  # default_sample_duration
        struct.pack_into('>I', data, last + 16, 0xfffffffb)
        cls.broken = cls.root / 'overflow.mp4'
        cls.broken.write_bytes(data)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_repairs_overflow_without_changing_frame_content_or_timestamps(self):
        self.assertGreater(service._probe_duration(str(self.broken)), 7_000_000)
        repaired, duration = service._prepare_chunk(str(self.broken))
        self.assertAlmostEqual(duration, 3, delta=0.1)
        before = media_json(self.broken, 'packet=stream_index,pts_time,data_hash')['packets']
        after = media_json(repaired, 'packet=stream_index,pts_time,data_hash')['packets']
        for index in [0, 1]:
            original = [p for p in before if p['stream_index'] == index]
            normalized = [p for p in after if p['stream_index'] == index]
            self.assertEqual([p['data_hash'] for p in original], [p['data_hash'] for p in normalized])
            for a, b in zip(original, normalized):
                self.assertAlmostEqual(float(a['pts_time']), float(b['pts_time']), delta=0.002)

    def test_concat_and_preview_use_the_same_repaired_timeline(self):
        reversed_tracks = self.root / 'audio-first.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(self.clean), '-map', '0:a', '-map', '0:v',
                        '-c', 'copy', str(reversed_tracks)], check=True, capture_output=True)
        with patch.object(service, 'SESS_DIR', str(self.root)):
            session = service.Session('concat', SimpleNamespace())
            session._maybe_dispatch = lambda: None
            for i, source in enumerate([self.clean, self.broken, reversed_tracks]):
                session.add_chunk(i, str(source), float('inf'))  # client duration must never override actual media
            self.assertAlmostEqual(session.seconds_received(), 9, delta=0.2)
            for chunk in session.chunks:
                streams = media_json(chunk['path'], 'stream=codec_type')['streams']
                self.assertEqual([s['codec_type'] for s in streams], ['video', 'audio'])
            window, _, total, chunks = session._build_window(session.chunks, 0)
            self.assertAlmostEqual(service._probe_duration(window), total, delta=0.1)
            self.assertEqual(service._video_segments(chunks, 4), [{'index': 1, 'offset_s': round(4 - chunks[0]['duration'], 6),
                                                                 'duration_s': 1.0}])

    def test_invalid_media_never_enters_session_timeline(self):
        bad = self.root / 'invalid.mp4'
        bad.write_bytes(b'not a video')
        with patch.object(service, 'SESS_DIR', str(self.root)):
            session = service.Session('invalid', SimpleNamespace())
            with self.assertRaises(ValueError):
                session.add_chunk(0, str(bad), 3)
            self.assertEqual(session.chunks, [])
            self.assertEqual(session.seconds_received(), 0)


class FinishTests(unittest.TestCase):
    def test_concurrent_finish_requests_share_one_final_window_and_cached_summary(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(service, 'SESS_DIR', directory):
            cls = service.TribeService._get_user_cls()
            svc = cls()
            session = service.Session('finish-test', svc)
            svc.session = lambda sid, create=False: session
            svc.summarize = lambda secs, arr: {'systems': [{'id': 'motion'}], 'timeline': [], 'n_seconds_predicted': 1}
            session.chunks = [{'index': 0, 't_start': 0, 'duration': 3}]
            entered, release = threading.Event(), threading.Event()

            def run_final(chunks, number, final):
                entered.set()
                self.assertTrue(release.wait(3))
                with session.lock:
                    session.inflight -= 1

            session._run_window = Mock(side_effect=run_final)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    responses = list(pool.map(lambda _: svc.finish(session.sid, wait=False), range(4)))
                self.assertTrue(entered.wait(1))
                self.assertTrue(all(r['state'] == 'finishing' for r in responses))
                self.assertFalse(session.pending)
                api = cls.web._get_raw_f()(svc)
                with TestClient(api) as client:
                    response = client.post('/session/finish-test/finish?wait=false')
                    self.assertEqual(response.status_code, 202)
                    release.set()
                    self.assertTrue(session.finish_done.wait(2))
                    a = client.post('/session/finish-test/finish?wait=false')
                    b = client.post('/session/finish-test/finish')
                    self.assertEqual(a.status_code, 200)
                    self.assertEqual(a.json(), b.json())
                    self.assertEqual(a.json()['status']['finish_state'], 'done')
                session._run_window.assert_called_once()
            finally:
                release.set()

    def test_laptop_survives_disconnect_and_more_than_120_seconds_of_finishing(self):
        elapsed = [0]
        requests = []
        summary = {'systems': [{'id': 'motion'}], 'n_seconds_predicted': 80, 'duration_s': 80}

        def respond(request):
            requests.append(request)
            if len(requests) == 1:
                raise httpx.ReadTimeout('dropped connection', request=request)
            if len(requests) < 8:
                return httpx.Response(202, json={'state': 'finishing', 'status': {'inflight': 1, 'seconds_predicted': 60}})
            return httpx.Response(200, json=summary)

        client = httpx.Client(transport=httpx.MockTransport(respond))
        log = Mock()
        with patch.object(server, '_modal_base_url', return_value='https://modal.test'), \
             patch.object(server.httpx, 'Client', return_value=client), \
             patch.object(server.time, 'monotonic', side_effect=lambda: elapsed[0]), \
             patch.object(server.time, 'sleep', side_effect=lambda _: elapsed.__setitem__(0, elapsed[0] + 30)):
            result = server._fetch_summary('test', log)
        self.assertGreater(elapsed[0], 120)
        self.assertEqual(result['n_seconds_predicted'], 80)
        self.assertTrue(all(r.url.params['wait'] == 'false' for r in requests))
        self.assertTrue(any('reconnecting' in call.args[0] for call in log.call_args_list))


if __name__ == '__main__':
    unittest.main()
