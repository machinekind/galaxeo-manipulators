"""GraspGenX ZMQ client against an in-process REP stub."""

from __future__ import annotations

import threading

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("zmq")
pytest.importorskip("msgpack")
pytest.importorskip("msgpack_numpy")

from galaxeo.vision.graspgenx import GraspGenXClient, SweepVolumeParams  # noqa: E402


def _rep_server(port, handler, ready):
    import msgpack
    import msgpack_numpy
    import zmq
    msgpack_numpy.patch()
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{port}")
    ready.set()
    raw = sock.recv()
    req = msgpack.unpackb(raw, raw=False)
    sock.send(msgpack.packb(handler(req), use_bin_type=True))
    sock.close()
    ctx.term()


def test_sweep_volume_from_gripper_config_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("""{
      "type": "revolute_2f",
      "fingertip": [0.0, 0.0, 0.12],
      "sweep_volume": {
        "extents": [0.08, 0.02, 0.04], "offset": [0.0, 0.0, 0.10],
        "extents2": [0.04, 0.02, 0.04], "offset2": [0.0, 0.0, 0.10]
      }
    }""")
    sv = SweepVolumeParams.from_gripper_config_json(str(p))
    assert sv.gripper_type == 1
    assert sv.fingertip_depth == pytest.approx(0.12)
    assert sv.extents_open[0] == pytest.approx(0.08)


def test_infer_sends_the_cloud_and_galaxea_g1(unused_tcp_port):
    pc = np.zeros((12, 3), np.float32)
    pc[:, 2] = 0.1
    grasps = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    grasps[0, :3, 3] = [0.3, 0, 0.1]
    scores = np.array([0.9, 0.4], np.float32)

    def handler(req):
        assert req["action"] == "infer"
        assert req["gripper_name"] == "galaxea_g1"
        assert np.asarray(req["point_cloud"]).shape == (12, 3)
        return {"grasps": grasps, "confidences": scores, "gripper_name": "galaxea_g1"}

    ready = threading.Event()
    t = threading.Thread(target=_rep_server, args=(unused_tcp_port, handler, ready),
                         daemon=True)
    t.start()
    assert ready.wait(2)
    with GraspGenXClient(host="127.0.0.1", port=unused_tcp_port, timeout_ms=5000) as c:
        g, s = c.infer(pc)
    t.join(2)
    assert g.shape == (2, 4, 4)
    assert s[0] == pytest.approx(0.9)


def test_infer_object_applies_topk_client_side(unused_tcp_port):
    params = SweepVolumeParams(
        extents_open=[0.08, 0.02, 0.04], offset_open=[0, 0, 0.1],
        extents_mid=[0.04, 0.02, 0.04], offset_mid=[0, 0, 0.1],
    )
    G = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    S = np.array([0.2, 0.9, 0.5], np.float32)

    def handler(req):
        assert req["action"] == "infer_object"
        assert "extents_open" in req["sweep_volume_params"]
        return {"grasps": G, "confidences": S}

    ready = threading.Event()
    t = threading.Thread(target=_rep_server, args=(unused_tcp_port, handler, ready),
                         daemon=True)
    t.start()
    assert ready.wait(2)
    pc = np.random.default_rng(0).normal(size=(20, 3)).astype(np.float32)
    with GraspGenXClient(host="127.0.0.1", port=unused_tcp_port, timeout_ms=5000) as c:
        g, s = c.infer_object(pc, params, topk_num_grasps=1, grasp_threshold=0.0)
    t.join(2)
    assert len(s) == 1 and s[0] == pytest.approx(0.9)


@pytest.fixture
def unused_tcp_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port
