"""Isaac Lab RGB camera configuration used by the HAIC VLA collector."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg


HAIC_VLA_CAMERA_PERIOD_S = 0.1
HAIC_VLA_CAMERA_WIDTH = 256
HAIC_VLA_CAMERA_HEIGHT = 256

# These are the D435 torso_link extrinsics already used by HAIC's depth ray
# camera.  Keep the world-frame convention so RGB and privileged collection
# observe the same robot-mounted viewpoint.
HAIC_D435_OFFSET_POS = (
    0.04764571478 + 0.0039635 - 0.0042 * math.cos(math.radians(48)),
    0.015,
    0.46268178553 - 0.044 + 0.0042 * math.sin(math.radians(48)) + 0.016,
)
HAIC_D435_OFFSET_ROT = (
    math.cos(math.radians(0.5) / 2) * math.cos(math.radians(48) / 2),
    math.sin(math.radians(0.5) / 2),
    math.sin(math.radians(48) / 2),
    0.0,
)


def haic_vla_camera_cfg() -> TiledCameraCfg:
    """Return the 256² RGB TiledCamera attached to ``torso_link``."""
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/vla_camera",
        update_period=HAIC_VLA_CAMERA_PERIOD_S,
        height=HAIC_VLA_CAMERA_HEIGHT,
        width=HAIC_VLA_CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.0,
            focus_distance=4.0,
            horizontal_aperture=2.0 * math.tan(math.radians(90.05) / 2.0),
            clipping_range=(0.1, 4.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=HAIC_D435_OFFSET_POS,
            rot=HAIC_D435_OFFSET_ROT,
            convention="world",
        ),
    )
