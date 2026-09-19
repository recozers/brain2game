"""CPU-only checks for the video/prediction timestamp contract. No Modal or model calls."""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from modal_app import tribe_service as service


class VideoPreviewTests(unittest.TestCase):
    def test_fractional_chunk_boundary(self):
        chunks = [{"index": 7, "duration": 3.2}, {"index": 8, "duration": 3.1}]
        self.assertEqual(service._video_segments(chunks, 3), [
            {"index": 7, "offset_s": 3.0, "duration_s": 0.2},
            {"index": 8, "offset_s": 0.0, "duration_s": 0.8},
        ])

    def test_exact_boundary_and_partial_final_second(self):
        chunks = [{"index": 4, "duration": 3}, {"index": 5, "duration": 2.4}]
        self.assertEqual(service._video_segments(chunks, 3), [
            {"index": 5, "offset_s": 0.0, "duration_s": 1.0},
        ])
        self.assertEqual(service._video_segments(chunks, 5), [
            {"index": 5, "offset_s": 2.0, "duration_s": 0.4},
        ])
        self.assertEqual(service._video_segments(chunks, 6), [])

    def test_video_provenance_follows_first_accepted_prediction(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(service, 'SESS_DIR', directory):
            svc = SimpleNamespace(
                infer_remote=lambda *a, **k: (np.ones((1, service.N_VERT)), np.array([0.]), {}),
                stats_for=lambda vec: ({"motion": 1.0}, []),
            )
            session = service.Session('test', svc)
            # Two overlapping windows complete with the same absolute second but different footage.
            for index in [7, 8]:
                path = Path(directory) / f'window{index}.mp4'
                path.write_bytes(b'not decoded by this test')
                chunks = [{"index": index, "duration": 3.2}]
                session._build_window = lambda *args: (str(path), 10.4, 3.2, chunks)
                session.inflight = 1
                session._run_window(chunks, index, True)
                self.assertIsNone(session.last_error)
            response = session.preds_since(10)
            self.assertEqual(response['seconds'], [10])
            self.assertEqual(response['video'], [{"second": 10, "clips": [
                {"index": 7, "offset_s": 0.0, "duration_s": 1.0},
            ]}])
            self.assertEqual(session.preds_since(11)['video'], [])

    def test_media_route_supports_seeking_and_missing_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chunk.mp4'
            path.write_bytes(b'0123456789')
            session = SimpleNamespace(lock=threading.Lock(), by_index={3: {"path": str(path)}})
            svc = SimpleNamespace(session=lambda sid, create=False: session if sid == 'test' else None)
            # Instantiate just the ASGI factory; bypass Modal's GPU/container lifecycle.
            api = service.TribeService._get_user_cls().web._get_raw_f()(svc)
            with TestClient(api) as client:
                response = client.get('/session/test/video/3', headers={"Range": "bytes=2-5"})
                self.assertEqual(response.status_code, 206)
                self.assertEqual(response.content, b'2345')
                self.assertEqual(response.headers['content-type'], 'video/mp4')
                self.assertEqual(response.headers['content-range'], 'bytes 2-5/10')
                self.assertEqual(response.headers['cache-control'], 'no-store')
                self.assertEqual(client.get('/session/test/video/4').status_code, 404)
                self.assertEqual(client.get('/session/missing/video/3').status_code, 404)


if __name__ == '__main__':
    unittest.main()
