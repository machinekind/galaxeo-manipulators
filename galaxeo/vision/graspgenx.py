"""Torch-free GraspGenX ZMQ client (msgpack + REQ/REP).

Wire format matches GraspGenX `client-server/README.md`: numpy arrays travel
through msgpack-numpy. The server owns the model; this side only sends a
cloud and a gripper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

FLAT_FIELD_ORDER = ("extents_open", "offset_open", "extents_mid", "offset_mid")
GRIPPER_TYPE_MAP = {"parallel_2f": 0, "revolute_2f": 1, "revolute_3f": 2}


@dataclass
class SweepVolumeParams:
    """12 numbers that condition GraspGenX when the server has no gripper assets."""

    extents_open: np.ndarray
    offset_open: np.ndarray
    extents_mid: np.ndarray
    offset_mid: np.ndarray
    gripper_type: int = 0
    fingertip_depth: Optional[float] = None

    def __post_init__(self) -> None:
        for name in FLAT_FIELD_ORDER:
            v = np.asarray(getattr(self, name), np.float32).reshape(-1)
            if v.shape != (3,):
                raise ValueError(f"{name} must have 3 elements; got {v.shape}")
            setattr(self, name, v)
        self.gripper_type = int(self.gripper_type)

    def to_dict(self) -> dict:
        d = {name: getattr(self, name) for name in FLAT_FIELD_ORDER}
        d["gripper_type"] = self.gripper_type
        if self.fingertip_depth is not None:
            d["fingertip_depth"] = float(self.fingertip_depth)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SweepVolumeParams":
        return cls(
            extents_open=d["extents_open"],
            offset_open=d["offset_open"],
            extents_mid=d["extents_mid"],
            offset_mid=d["offset_mid"],
            gripper_type=d.get("gripper_type", 0),
            fingertip_depth=d.get("fingertip_depth"),
        )

    @classmethod
    def from_gripper_config_json(cls, path: str) -> "SweepVolumeParams":
        """`config.json` from a GraspGenX gripper_descriptions asset."""
        import json
        with open(path) as f:
            config = json.load(f)
        sv = config["sweep_volume"]
        return cls(
            extents_open=sv["extents"],
            offset_open=sv["offset"],
            extents_mid=sv["extents2"],
            offset_mid=sv["offset2"],
            gripper_type=GRIPPER_TYPE_MAP.get(config.get("type"), 0),
            fingertip_depth=config.get("fingertip", [None, None, None])[-1],
        )


class GraspGenXClient:
    """REQ client. Default gripper name is the G1 on this arm."""

    def __init__(self, host: str = "localhost", port: int = 5556,
                 timeout_ms: Optional[int] = 60_000):
        self.host = host
        self.port = int(port)
        self.timeout_ms = timeout_ms
        self._ctx = None
        self._sock = None

    @property
    def address(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def __enter__(self) -> "GraspGenXClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def connect(self) -> None:
        if self._sock is not None:
            return
        import zmq
        self._ctx = zmq.Context.instance()
        sock = self._ctx.socket(zmq.REQ)
        if self.timeout_ms is not None:
            sock.setsockopt(zmq.RCVTIMEO, int(self.timeout_ms))
            sock.setsockopt(zmq.SNDTIMEO, int(self.timeout_ms))
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.address)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None

    def _request(self, payload: dict) -> dict:
        import msgpack
        import msgpack_numpy
        import zmq
        msgpack_numpy.patch()
        if self._sock is None:
            self.connect()
        try:
            self._sock.send(msgpack.packb(payload, use_bin_type=True))
            raw = self._sock.recv()
        except zmq.error.Again as exc:
            self.close()
            raise TimeoutError(
                f"GraspGenX at {self.address} did not reply in {self.timeout_ms} ms"
            ) from exc
        response = msgpack.unpackb(raw, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GraspGenX: {response['error']}")
        return response

    def health(self) -> dict:
        return self._request({"action": "health"})

    def metadata(self) -> dict:
        return self._request({"action": "metadata"})

    def infer(self, point_cloud: np.ndarray, gripper_name: str = "galaxea_g1",
              num_grasps: int = 200, grasp_threshold: float = -1.0,
              topk_num_grasps: int = 100):
        """Name-based infer: server loads gripper assets. Returns (grasps, scores)."""
        pc = _pc(point_cloud)
        response = self._request({
            "action": "infer",
            "point_cloud": pc,
            "gripper_name": gripper_name,
            "num_grasps": int(num_grasps),
            "grasp_threshold": float(grasp_threshold),
            "topk_num_grasps": int(topk_num_grasps),
        })
        return (np.asarray(response["grasps"], np.float32),
                np.asarray(response["confidences"], np.float32))

    def infer_object(self, point_cloud: np.ndarray,
                     sweep_volume_params: Union[SweepVolumeParams, dict],
                     planner: str = "diffusion", num_grasps: int = 200,
                     grasp_threshold: float = 0.0, topk_num_grasps: int = 100):
        """Params-only object cloud. Threshold/top-k applied here."""
        params = (sweep_volume_params if isinstance(sweep_volume_params, SweepVolumeParams)
                  else SweepVolumeParams.from_dict(sweep_volume_params))
        response = self._request({
            "action": "infer_object",
            "point_cloud": _pc(point_cloud),
            "sweep_volume_params": params.to_dict(),
            "planner": planner,
            "num_grasps": int(num_grasps),
        })
        grasps = np.asarray(response["grasps"], np.float32).reshape(-1, 4, 4)
        scores = np.asarray(response["confidences"], np.float32).reshape(-1)
        return _select(grasps, scores, grasp_threshold, topk_num_grasps)

    def infer_scene_pc(self, point_cloud: np.ndarray, instance_mask: np.ndarray,
                       sweep_volume_params: Union[SweepVolumeParams, dict],
                       planner: str = "diffusion", num_grasps: int = 200,
                       min_object_points: int = 100,
                       grasp_threshold: float = 0.0, topk_num_grasps: int = 100):
        """Scene cloud + instance ids (0 = ignore) → {id: (grasps, scores)}."""
        params = (sweep_volume_params if isinstance(sweep_volume_params, SweepVolumeParams)
                  else SweepVolumeParams.from_dict(sweep_volume_params))
        pc = np.asarray(point_cloud, np.float32)
        mask = np.asarray(instance_mask)
        response = self._request({
            "action": "infer_scene_pc",
            "point_cloud": pc,
            "instance_mask": mask,
            "sweep_volume_params": params.to_dict(),
            "planner": planner,
            "num_grasps": int(num_grasps),
            "min_object_points": int(min_object_points),
        })
        out = {}
        ids = np.asarray(response["instance_ids"]).reshape(-1)
        for i, inst in enumerate(ids.tolist()):
            g, s = _select(
                np.asarray(response["grasps"][i], np.float32).reshape(-1, 4, 4),
                np.asarray(response["confidences"][i], np.float32).reshape(-1),
                grasp_threshold, topk_num_grasps,
            )
            if len(g):
                out[int(inst)] = (g, s)
        return out


def _pc(point_cloud: np.ndarray) -> np.ndarray:
    pc = np.asarray(point_cloud, np.float32)
    if pc.ndim != 2 or pc.shape[1] != 3 or len(pc) == 0:
        raise ValueError(f"point_cloud must be non-empty (N, 3); got {pc.shape}")
    return pc


def _select(grasps, scores, threshold, topk):
    if threshold > 0:
        keep = scores >= threshold
        grasps, scores = grasps[keep], scores[keep]
    if topk is not None and topk > 0 and len(scores):
        order = np.argsort(-scores)[:topk]
        grasps, scores = grasps[order], scores[order]
    return grasps, scores
