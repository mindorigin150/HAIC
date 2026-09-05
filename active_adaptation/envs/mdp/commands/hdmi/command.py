from active_adaptation.envs.mdp.base import Command
from active_adaptation.utils.motion import MotionDataset, MotionData

from typing import List, Dict, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor
    from isaaclab.assets import Articulation, RigidObject

import torch
import numpy as np
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz, quat_mul, quat_apply, quat_apply_inverse
from tensordict import TensorDict
from active_adaptation.utils.math import batchify
quat_apply = batchify(quat_apply)
quat_apply_inverse = batchify(quat_apply_inverse)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)

class RobotTracking(Command):
    def __init__(
        self, env, data_path: List[str] | str,
        tracking_keypoint_names: List[str],
        tracking_joint_names: List[str],
        # reset parameters
        root_body_name: str = "pelvis",
        reset_range: Tuple[float, float] | None = None,
        pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        velocity_range: Dict[str, Tuple[float, float]] = {
            "x": (-0., 0.),
            "y": (-0., 0.),
            "z": (-0., 0.),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        init_joint_pos_noise: float = 0.0,
        init_joint_vel_noise: float = 0.0,
        # observation parameters
        future_steps: List[int] = [1, 2, 8, 16],
        call_update: bool = True,
        sample_motion: bool = False,
        replay_motion: bool = False,
        record_motion: bool = False,
    ):
        from . import observations
        from . import rewards
        from . import randomizations
        from . import terminations
        super().__init__(env)
        self.dataset = MotionDataset.create_from_path(
            data_path,
            isaac_joint_names=self.asset.joint_names,
            target_fps=int(1/self.env.step_dt)
        ).to(self.device)

        # Set tracking body and joint names for observation and termination
        self.tracking_keypoint_names = self.asset.find_bodies(tracking_keypoint_names)[1]
        self.tracking_body_indices_motion = [self.dataset.body_names.index(name) for name in self.tracking_keypoint_names]
        self.tracking_body_indices_asset = [self.asset.body_names.index(name) for name in self.tracking_keypoint_names]

        self.tracking_joint_names = self.asset.find_joints(tracking_joint_names)[1]
        self.tracking_joint_indices_motion = [self.dataset.joint_names.index(name) for name in self.tracking_joint_names]
        self.tracking_joint_indices_asset = [self.asset.joint_names.index(name) for name in self.tracking_joint_names]

        self.num_tracking_bodies = len(self.tracking_body_indices_asset)
        self.num_tracking_joints = len(self.tracking_joint_indices_asset)
        self.num_future_steps = len(future_steps)

        # get root body and joint indices in motion for reset
        self.root_body_name = root_body_name
        self.root_body_idx_motion = self.dataset.body_names.index(root_body_name)
        
        asset_joint_names = self.asset.joint_names
        self.asset_joint_idx_motion = [self.dataset.joint_names.index(joint_name) for joint_name in asset_joint_names]

        with torch.device(self.device):
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.future_steps = torch.tensor(future_steps)

            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_len = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_starts = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_ends = torch.zeros(self.num_envs, dtype=torch.long)
            self.t = torch.zeros(self.num_envs, dtype=torch.long)
            self.replay_motion_t = torch.zeros(self.num_envs, dtype=torch.long)

            self.eval_t = torch.randint(0, self.dataset.lengths[0], (self.num_envs,), device=self.device)

        self.reset_range = reset_range

        pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.pose_range = torch.tensor(pose_range_list, device=self.device)
        velocity_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.velocity_range = torch.tensor(velocity_range_list, device=self.device)

        self.init_joint_pos_noise = init_joint_pos_noise
        self.init_joint_vel_noise = init_joint_vel_noise

        self.first_sample_motion = True
        self.sample_motion = sample_motion
        self.replay_motion = replay_motion
        self.record_motion = record_motion

        if self.replay_motion:
            self.pose_range.fill_(0.0)
            self.init_joint_pos_noise = 0.0
            self.init_joint_vel_noise = 0.0
        
        if self.record_motion:
            assert self.num_envs == 1, "record_motion only supports num_envs=1"
            self.pose_range.fill_(0.0)
            self.init_joint_pos_noise = 0.0
            self.init_joint_vel_noise = 0.0

        if call_update:
            self._init_debug_draw()
            self.update()
            if self.record_motion:
                self.motion_frames = []
        
    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        if self.sample_motion or self.first_sample_motion:
            # sample motion id and start time for each env
            if self.first_sample_motion:
                # round-robin assignment to ensure uniform distribution across motions
                motion_ids = torch.arange(len(env_ids), device=self.device) % self.dataset.num_motions
            else:
                motion_ids = self.env.random_int(
                    0,
                    self.dataset.num_motions,
                    (len(env_ids),),
                    env_ids=env_ids,
                )
            self.motion_ids[env_ids] = motion_ids
            self.motion_len[env_ids] = motion_len = self.dataset.lengths[motion_ids]
            self.motion_starts[env_ids] = self.dataset.starts[motion_ids]
            self.motion_ends[env_ids] = self.dataset.ends[motion_ids]
            self.first_sample_motion = False
        else:
            motion_len = self.motion_len[env_ids]

        if self.reset_range is None:
            max_len = motion_len - self.future_steps[-1]
            start_phase = self.env.random_uniform(
                0.0,
                1.0,
                (len(env_ids),),
                env_ids=env_ids,
            )
            start_t = (start_phase * max_len).long()
        else:
            start_t = self.env.random_int(
                self.reset_range[0],
                self.reset_range[1],
                (len(env_ids),),
                env_ids=env_ids,
            )
            
        if not self.env.training or self.record_motion:
            start_t.fill_(0)

        if self.replay_motion:
            self.replay_motion_t[env_ids] = (self.replay_motion_t[env_ids] + 1) % motion_len
            start_t = self.replay_motion_t[env_ids]

        self.t[env_ids] = start_t


    def sample_init(self, env_ids: torch.Tensor) -> None:
        self._sample_motions(env_ids)

        # reset root state and joint position/velocity from motion
        self._motion_reset: MotionData = self.dataset.get_slice(self.motion_ids[env_ids], self.t[env_ids], 1).squeeze(1)
        # shape: [len(env_ids), num_bodies/num_joints, 3/4/...]
        
        motion = self._motion_reset
        init_root_pos = motion.body_pos_w[:, self.root_body_idx_motion]
        init_root_quat = motion.body_quat_w[:, self.root_body_idx_motion]
        init_root_lin_vel = motion.body_lin_vel_w[:, self.root_body_idx_motion]
        init_root_ang_vel = motion.body_ang_vel_w[:, self.root_body_idx_motion]

        # poses
        rand_samples = self.env.random_uniform(
            self.pose_range[:, 0],
            self.pose_range[:, 1],
            (len(env_ids), 6),
            env_ids=env_ids,
        )
        if not self.env.training:
            rand_samples.fill_(0.0)
        init_root_pos = init_root_pos + self.env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        init_root_quat = quat_mul(init_root_quat, orientations_delta)

        # velocities
        rand_samples = self.env.random_uniform(
            self.velocity_range[:, 0],
            self.velocity_range[:, 1],
            (len(env_ids), 6),
            env_ids=env_ids,
        )
        if not self.env.training:
            rand_samples.fill_(0.0)
        velocities = torch.cat([init_root_lin_vel, init_root_ang_vel], dim=-1) + rand_samples

        self.asset.write_root_link_pose_to_sim(torch.cat([init_root_pos, init_root_quat], dim=-1), env_ids=env_ids)
        self.asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)

        init_joint_pos = motion.joint_pos[:, self.asset_joint_idx_motion]
        init_joint_vel = motion.joint_vel[:, self.asset_joint_idx_motion]

        joint_pos_noise = self.env.random_uniform(
            -1,
            1,
            (init_joint_pos.shape[0], init_joint_pos.shape[1]),
            env_ids=env_ids,
        ) * self.init_joint_pos_noise
        joint_vel_noise = self.env.random_uniform(
            -1,
            1,
            (init_joint_vel.shape[0], init_joint_vel.shape[1]),
            env_ids=env_ids,
        ) * self.init_joint_vel_noise

        init_joint_pos += joint_pos_noise
        init_joint_vel += joint_vel_noise

        joint_pos_limits = self.asset.data.soft_joint_pos_limits[env_ids]
        joint_vel_limits = self.asset.data.soft_joint_vel_limits[env_ids]
        init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        self.asset.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=env_ids)

        if self.record_motion:
            if len(self.motion_frames) > 0:
                self._save_motion()
                self.motion_frames = []
    
    def _save_motion(self):
        motion_data: TensorDict = torch.cat(self.motion_frames, dim=0)
        motion_data = motion_data[25:].numpy()
        moton_meta = {
            "joint_names": self.asset.joint_names,
            "body_names": self.asset.body_names,
            "fps": int(1/self.env.step_dt),
        }
        save_dir = "record_motion"
        motion_data_path = f"{save_dir}/motion.npz"
        motion_meta_path = f"{save_dir}/meta.json"
        import os, json
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(motion_data_path, **motion_data)
        with open(motion_meta_path, "w") as f:
            json.dump(moton_meta, f, indent=4)
        print(f"Saved recorded motion to {motion_data_path} and {motion_meta_path}")
        breakpoint()
            

    @property
    def success(self):
        return (self.t >= self.motion_len - 1).unsqueeze(1)
    
    @property
    def finished(self):
        if self.replay_motion:
            return torch.ones(self.num_envs, 1, dtype=bool, device=self.device)
        return (self.t >= self.motion_len).unsqueeze(1)

    def update(self):
        if hasattr(self, "motion_frames"):
            motion_frame = {}
            motion_frame["body_pos_w"] = self.asset.data.body_link_pos_w.cpu()
            motion_frame["body_quat_w"] = self.asset.data.body_link_quat_w.cpu()
            motion_frame["body_lin_vel_w"] = self.asset.data.body_com_lin_vel_w.cpu()
            motion_frame["body_ang_vel_w"] = self.asset.data.body_com_ang_vel_w.cpu()
            motion_frame["joint_pos"] = self.asset.data.joint_pos.cpu()
            motion_frame["joint_vel"] = self.asset.data.joint_vel.cpu()
            self.motion_frames.append(TensorDict(motion_frame, batch_size=[1]))
            
        # future ref motion for actor observation
        self.future_ref_motion = self.dataset.get_slice(self.motion_ids, self.t, steps=self.future_steps)
        # shape: [num_envs, len(future_steps), num_bodies/num_joints, 3/4/...]

        # Observations: future ref body and joint states
        self.ref_body_pos_future_w = self.future_ref_motion.body_pos_w[..., self.tracking_body_indices_motion, :] + self.env.scene.env_origins[:, None, None, :]
        self.ref_body_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[..., self.tracking_body_indices_motion, :]
        self.ref_body_quat_future_w = self.future_ref_motion.body_quat_w[..., self.tracking_body_indices_motion, :]
        self.ref_body_ang_vel_future_w = self.future_ref_motion.body_ang_vel_w[..., self.tracking_body_indices_motion, :]
        self.ref_joint_pos_future_ = self.future_ref_motion.joint_pos[..., self.tracking_joint_indices_motion]
        self.ref_joint_vel_future_ = self.future_ref_motion.joint_vel[..., self.tracking_joint_indices_motion]
        self.ref_root_pos_future_w = self.future_ref_motion.body_pos_w[..., self.root_body_idx_motion, :] + self.env.scene.env_origins[:, None, :]
        self.ref_root_quat_future_w = self.future_ref_motion.body_quat_w[..., self.root_body_idx_motion, :]
        self.ref_root_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[..., self.root_body_idx_motion, :]
        self.ref_root_ang_vel_future_w = self.future_ref_motion.body_ang_vel_w[..., self.root_body_idx_motion, :]

        # Reward: current robot body and joint states
        self.robot_body_pos_w = self.asset.data.body_link_pos_w[:, self.tracking_body_indices_asset]
        self.robot_body_lin_vel_w = self.asset.data.body_com_lin_vel_w[:, self.tracking_body_indices_asset]
        self.robot_body_quat_w = self.asset.data.body_link_quat_w[:, self.tracking_body_indices_asset]
        self.robot_body_ang_vel_w = self.asset.data.body_com_ang_vel_w[:, self.tracking_body_indices_asset]
        self.robot_joint_pos = self.asset.data.joint_pos[:, self.tracking_joint_indices_asset]
        self.robot_joint_vel = self.asset.data.joint_vel[:, self.tracking_joint_indices_asset]
        self.robot_root_pos_w = self.asset.data.root_link_pos_w
        self.robot_root_quat_w = self.asset.data.root_link_quat_w

        # Reward: current ref body and joint states
        self.current_ref_motion: MotionData = self.future_ref_motion[:, 0]
        self.ref_body_pos_w = self.ref_body_pos_future_w[:, 0]
        self.ref_body_lin_vel_w = self.ref_body_lin_vel_future_w[:, 0]
        self.ref_body_quat_w = self.ref_body_quat_future_w[:, 0]
        self.ref_body_ang_vel_w = self.ref_body_ang_vel_future_w[:, 0]
        self.ref_joint_pos = self.ref_joint_pos_future_[:, 0]
        self.ref_joint_vel = self.ref_joint_vel_future_[:, 0]
        self.ref_root_pos_w = self.ref_root_pos_future_w[:, 0]
        self.ref_root_quat_w = self.ref_root_quat_future_w[:, 0]
        # shape: [num_envs, num_future_steps, num_tracking_bodies, xxx]

        if self.env.backend == "isaac":
            self.all_marker_pos_w[0] = self.robot_body_pos_w
            self.all_marker_pos_w[1] = self.ref_body_pos_w
            # self.all_marker_pos_w[0] = self.ref_body_pos_future_w[:, 0]
            # self.all_marker_pos_w[1] = self.ref_body_pos_future_w[:, -1]

        self.t += 1
    
    def _init_debug_draw(self):
        if self.env.backend != "isaac":
            return
        
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        vis_markers_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Keypoints",
            markers={
                "robot": sim_utils.SphereCfg(
                    radius=0.04,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0)
                    ),
                ),
                "reference": sim_utils.SphereCfg(
                    radius=0.04,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0)
                    ),
                ),
            },
        )
        self.vis_markers = VisualizationMarkers(vis_markers_cfg)
        num_ref_markers = self.num_envs * self.num_tracking_bodies
        self.marker_indices = [0] * num_ref_markers + [1] * num_ref_markers
        self.all_marker_pos_w = torch.zeros(2, self.num_envs, self.num_tracking_bodies, 3, device=self.device)

    def debug_draw(self):
        if self.env.backend != "isaac":
            return

        if self.replay_motion:
            self.all_marker_pos_w.fill_(-1000)
        
        # shape: [2, num_envs, num_tracking_bodies, 3]
        self.vis_markers.visualize(
            translations=self.all_marker_pos_w.reshape(-1, 3),
            marker_indices=self.marker_indices,
        )

        # robot_keypoints_w = self.all_marker_pos_w[0].reshape(-1, 3)
        # target_keypoints_w = self.all_marker_pos_w[1].reshape(-1, 3)
        # self.env.debug_draw.vector(
        #     robot_keypoints_w,
        #     target_keypoints_w - robot_keypoints_w,
        #     color=(0, 0, 1, 1)
        # )

class RobotObjectTracking(RobotTracking):
    def __init__(
        self,
        extra_object_names: List[str],
        object_asset_name: str, # for finding the object in the scene
        object_body_name: str, # for the body that defines the contact target position
        object_joint_name: List[str] | None = None, # object joint to track
        # for reset
        object_pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        ## per-motion overrides for pose_range and object_pose_range
        pose_range_per_motion: List[Dict[str, Tuple[float, float]]] | None = None,
        object_pose_range_per_motion: List[Dict[str, Tuple[float, float]]] | None = None,
        object_init_joint_pos_noise: float = 0.1, 
        object_init_joint_vel_noise: float = 0.1,
        # for contact rewards
        contact_eef_body_name: List[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        contact_frc_eef_body_name: List[str | List[str]] = ["left_wrist_(roll|pitch|yaw)_link", "right_wrist_(roll|pitch|yaw)_link"],
        ## offset from object to contact target position
        contact_target_pos_offset: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        ## offset from end effector
        contact_eef_pos_offset: List[Tuple[float, float, float]] = [(0.1, 0.0, 0.0), (0.1, 0.0, 0.0)],
        ## per-motion overrides: list of per-motion contact_target_pos_offset (one entry per motion in data_path)
        ## if provided, contact offsets are switched per-env when motions are sampled
        contact_target_pos_offset_per_motion: List[List[Tuple[float, float, float]]] | None = None,
        contact_eef_pos_offset_per_motion: List[List[Tuple[float, float, float]]] | None = None,
        object2_asset_name: str | None = None,
        object2_body_name: str | None = None,
        object2_joint_name: List[str] | None = None,
        object2_pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        contact2_eef_body_name: List[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        contact2_target_pos_offset: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        contact2_eef_pos_offset: List[Tuple[float, float, float]] = [(0.1, 0.0, 0.0), (0.1, 0.0, 0.0)],
        **kwargs
    ):
        super().__init__(**kwargs, call_update=False)

        self.extra_objects: List[Articulation | RigidObject] = [self.env.scene[name] for name in extra_object_names]
        self.extra_object_body_id_motion = [self.dataset.body_names.index(name) for name in extra_object_names]

        self.object_asset_name = object_asset_name
        if object_joint_name is None:
            self.object = self.env.scene.rigid_objects[object_asset_name]
            self.object_joint_idx_motion = None
            self.object_joint_idx_asset = None
        else:
            self.object = self.env.scene.articulations[object_asset_name]
            self.object_joint_idx_motion = [self.dataset.joint_names.index(j) for j in object_joint_name]
            self.object_joint_idx_asset = [self.object.joint_names.index(j) for j in object_joint_name]
        
        self.object_body_id_asset = self.object.body_names.index(object_body_name)
        self.object_body_id_motion = self.dataset.body_names.index(object_asset_name)

        self.object2_asset_name = object2_asset_name
        if object2_asset_name is None:
            self.object2 = None
        else:
            if object2_joint_name is None:
                self.object2 = self.env.scene.rigid_objects[object2_asset_name]
                self.object2_joint_idx_motion = None
                self.object2_joint_idx_asset = None
            else:
                self.object2 = self.env.scene.articulations[object2_asset_name]
                self.object2_joint_idx_motion = [self.dataset.joint_names.index(j) for j in object2_joint_name]
                self.object2_joint_idx_asset = [self.object2.joint_names.index(j) for j in object2_joint_name]
            
            self.object2_body_id_asset = self.object2.body_names.index(object2_body_name)
            self.object2_body_id_motion = self.dataset.body_names.index(object2_asset_name)

        pose_range_list = [object_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.object_pose_range = torch.tensor(pose_range_list, device=self.device)
        self.object_init_joint_pos_noise = object_init_joint_pos_noise
        self.object_init_joint_vel_noise = object_init_joint_vel_noise

        if self.replay_motion or self.record_motion:
            self.object_pose_range.fill_(0.0)
            self.object_init_joint_pos_noise = 0.0
            self.object_init_joint_vel_noise = 0.0

        # per-motion pose_range and object_pose_range tables: shape (num_motions, 6, 2)
        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        if pose_range_per_motion is not None:
            self._pose_range_per_motion = torch.tensor(
                [[pr.get(k, (0.0, 0.0)) for k in keys] for pr in pose_range_per_motion],
                dtype=torch.float32, device=self.device
            )  # (num_motions, 6, 2)
        if object_pose_range_per_motion is not None:
            self._object_pose_range_per_motion = torch.tensor(
                [[pr.get(k, (0.0, 0.0)) for k in keys] for pr in object_pose_range_per_motion],
                dtype=torch.float32, device=self.device
            )  # (num_motions, 6, 2)

        if self.object2 is not None:    
            pose_range_list = [object2_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            self.object2_pose_range = torch.tensor(pose_range_list, device=self.device)
            self.object2_init_joint_pos_noise = object_init_joint_pos_noise
            self.object2_init_joint_vel_noise = object_init_joint_vel_noise

            if self.replay_motion or self.record_motion:
                self.object2_pose_range.fill_(0.0)
                self.object2_init_joint_pos_noise = 0.0
                self.object2_init_joint_vel_noise = 0.0

        # setup contact body indices
        assert len(contact_eef_body_name) == len(contact_target_pos_offset) == len(contact_eef_pos_offset), \
            "contact_eef_body_name, contact_target_pos_offset, and contact_eef_pos_offset must have the same length"
        self.num_eefs = len(contact_eef_body_name)
        self.contact_eef_body_indices_asset = [self.asset.body_names.index(name) for name in contact_eef_body_name]

        self.eef_filtered_sensor: List[List[ContactSensor]] = []
        # [self.env.scene.sensors[f"{eef_name}_{object_asset_name}_contact_forces"] for eef_name in contact_eef_body_name] for object_asset_name in self.asset.data.object_names]
        self.eef_filtered_sensor_indices: List[List[int]] = []
        # = [eef_sensor.body_names.index(eef_name) for (eef_name, eef_sensor) in zip(contact_eef_body_name, self.eef_object_contact_forces)]
        for eef_name in contact_eef_body_name:
            eef_names = self.asset.find_bodies(eef_name)[1]
            sensors_for_this_eef = []
            sensor_indices_for_this_eef = []
            for eef_name in eef_names:
                eef_sensor_name = f"{eef_name}_{object_asset_name}_contact_forces"
                eef_sensor_filtered = self.env.scene.sensors[eef_sensor_name]
                sensors_for_this_eef.append(eef_sensor_filtered)
                sensor_indices_for_this_eef.append(eef_sensor_filtered.body_names.index(eef_name))
            self.eef_filtered_sensor.append(sensors_for_this_eef)
            self.eef_filtered_sensor_indices.append(sensor_indices_for_this_eef)
        
        if self.object2 is not None:
            assert len(contact2_eef_body_name) == len(contact2_target_pos_offset) == len(contact2_eef_pos_offset), \
                "contact2_eef_body_name, contact2_target_pos_offset, and contact2_eef_pos_offset must have the same length"
            self.num_eefs2 = len(contact2_eef_body_name)
            self.contact2_eef_body_indices_asset = [self.asset.body_names.index(name) for name in contact2_eef_body_name]

            self.eef2_filtered_sensor: List[List[ContactSensor]] = []
            self.eef2_filtered_sensor_indices: List[List[int]] = []
            for eef_name in contact2_eef_body_name:
                eef_names = self.asset.find_bodies(eef_name)[1]
                sensors_for_this_eef = []
                sensor_indices_for_this_eef = []
                for eef_name in eef_names:
                    eef_sensor_name = f"{eef_name}_{object2_asset_name}_contact_forces"
                    eef_sensor_filtered = self.env.scene.sensors[eef_sensor_name]
                    sensors_for_this_eef.append(eef_sensor_filtered)
                    sensor_indices_for_this_eef.append(eef_sensor_filtered.body_names.index(eef_name))
                self.eef2_filtered_sensor.append(sensors_for_this_eef)
                self.eef2_filtered_sensor_indices.append(sensor_indices_for_this_eef)

        with torch.device(self.device):
            self.contact_target_pos_offset = torch.tensor(contact_target_pos_offset, device=self.device).repeat(self.num_envs, 1, 1)
            self.contact_eef_pos_offset = torch.tensor(contact_eef_pos_offset, device=self.device).repeat(self.num_envs, 1, 1)

            self.contact_target_pos_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
            self.contact_eef_pos_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)

            self.eef_contact_forces_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
            self.eef_contact_forces_b = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)

        # per-motion contact offset tables: shape (num_motions, num_eefs, 3)
        if contact_target_pos_offset_per_motion is not None:
            self._contact_target_pos_offset_per_motion = torch.tensor(
                contact_target_pos_offset_per_motion, dtype=torch.float32, device=self.device
            )  # (num_motions, num_eefs, 3)
        if contact_eef_pos_offset_per_motion is not None:
            self._contact_eef_pos_offset_per_motion = torch.tensor(
                contact_eef_pos_offset_per_motion, dtype=torch.float32, device=self.device
            )  # (num_motions, num_eefs, 3)

            if self.object2 is not None:
                self.contact2_target_pos_offset = torch.tensor(contact2_target_pos_offset, device=self.device).repeat(self.num_envs, 1, 1)
                self.contact2_eef_pos_offset = torch.tensor(contact2_eef_pos_offset, device=self.device).repeat(self.num_envs, 1, 1)

                self.contact2_target_pos_w = torch.zeros(self.num_envs, len(contact2_eef_body_name), 3, device=self.device)
                self.contact2_eef_pos_w = torch.zeros(self.num_envs, len(contact2_eef_body_name), 3, device=self.device)

                self.eef2_contact_forces_w = torch.zeros(self.num_envs, len(contact2_eef_body_name), 3, device=self.device)
                self.eef2_contact_forces_b = torch.zeros(self.num_envs, len(contact2_eef_body_name), 3, device=self.device)

        scale = getattr(self.object.cfg.spawn, "scale", None)
        if not isinstance(scale, torch.Tensor):
            scale_tensor = torch.ones(self.num_envs, 3)
            if scale is None:
                pass
            elif isinstance(scale, float):
                scale_tensor[:] = scale
            elif isinstance(scale, tuple):
                scale_tensor[:] = torch.tensor(scale)
            else:
                raise ValueError(f"Invalid scale type: {type(scale)}")
            scale = scale_tensor
        self.contact_target_pos_offset *= scale.unsqueeze(1).to(self.device)
        if self.object2 is not None:
            self.contact2_target_pos_offset *= scale.unsqueeze(1).to(self.device)   

        # load object contact data (supports multiple motion paths)
        motion_paths = self.dataset.motion_paths
        contact_eef_body_idx = [self.dataset.contact_body_names.index(name) for name in contact_eef_body_name]
        if self.object2 is not None:
            contact2_eef_body_idx = [self.dataset.contact_body_names.index(name) for name in contact2_eef_body_name]

        all_object_contact = []
        all_body_contact = []
        all_feet_contact = []
        all_body2_contact = []
        first_motion_data = None
        for path in motion_paths:
            motion_data = np.load(path, allow_pickle=True)
            if first_motion_data is None:
                first_motion_data = motion_data
            all_object_contact.append(motion_data["object_contact"])
            all_body_contact.append(motion_data["body_contact"][:, contact_eef_body_idx])
            all_feet_contact.append(motion_data["feet_contact"])
            if self.object2 is not None:
                all_body2_contact.append(motion_data["body_contact"][:, contact2_eef_body_idx])

        object_contact = np.concatenate(all_object_contact, axis=0)
        body_contact = np.concatenate(all_body_contact, axis=0)
        feet_contact = np.concatenate(all_feet_contact, axis=0)
        object_points = first_motion_data["object_points"]  # [1, 128, 3] — same box shape for all motions

        self._object_contact = torch.tensor(object_contact[...,:1], device=self.device, dtype=torch.bool)
        self._body_contact = torch.tensor(body_contact, device=self.device, dtype=torch.bool) & self._object_contact
        self._feet_contact = torch.tensor(feet_contact, device=self.device, dtype=torch.bool)
        self.object_points = torch.tensor(object_points[:1], device=self.device).repeat(self.num_envs, 1, 1)

        if self.object2 is not None:
            body2_contact = np.concatenate(all_body2_contact, axis=0)
            self._object2_contact = torch.tensor(object_contact[...,1:2], device=self.device, dtype=torch.bool)
            self._body2_contact = torch.tensor(body2_contact, device=self.device, dtype=torch.bool) & self._object2_contact
            self.object2_points = torch.tensor(object_points[1:2], device=self.device).repeat(self.num_envs, 1, 1)

        if len(self.extra_objects) > 0:
            self.extra_object_points = torch.tensor(object_points[-1:], device=self.device).repeat(self.num_envs, 1, 1)

        # per-motion mask: True if the motion has a stairslope extra object (all motions except box_1)
        self._motion_has_extra_object = torch.tensor(
            ["box_1" not in str(path) for path in motion_paths],
            dtype=torch.bool, device=self.device
        )
        self.has_extra_object_mask = self._motion_has_extra_object[self.motion_ids]  # shape: (num_envs,)

        # if self._object_contact.shape[1] == 1:
        #     # expand to num_eefs
        #     self._object_contact = self._object_contact.repeat(1, self.num_eefs)
        # # shape: [num_steps, num_eefs/1]

        self._init_debug_draw()

        # Park extra objects for envs with no extra object (e.g. box_1) before first update
        if len(self.extra_objects) > 0 and hasattr(self, "_motion_has_extra_object"):
            no_extra_envs = (~self._motion_has_extra_object[self.motion_ids]).nonzero(as_tuple=False).squeeze(-1)
            if len(no_extra_envs) > 0:
                park_pos = self.env.scene.env_origins[no_extra_envs].clone()
                park_pos[:, 2] -= 1000.0
                for object_ in self.extra_objects:
                    state = object_.data.default_root_state[no_extra_envs].clone()
                    state[:, 0:3] = park_pos
                    state[:, 7:] = 0.0
                    object_.write_root_link_pose_to_sim(state[:, 0:7], env_ids=no_extra_envs)
                    object_.write_root_com_velocity_to_sim(state[:, 7:], env_ids=no_extra_envs)

        self.update()
        if self.record_motion:
            self.motion_frames = []

    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        super()._sample_motions(env_ids)
        if hasattr(self, "_motion_has_extra_object"):
            self.has_extra_object_mask[env_ids] = self._motion_has_extra_object[self.motion_ids[env_ids]]
            # Park extra objects far away for envs whose motion has no extra object (e.g. box_1)
            if len(self.extra_objects) > 0:
                no_extra = env_ids[~self.has_extra_object_mask[env_ids]]
                if len(no_extra) > 0:
                    park_pos = self.env.scene.env_origins[no_extra].clone()
                    park_pos[:, 2] -= 1000.0
                    for object_ in self.extra_objects:
                        state = object_.data.default_root_state[no_extra].clone()
                        state[:, 0:3] = park_pos
                        state[:, 7:] = 0.0
                        object_.write_root_link_pose_to_sim(state[:, 0:7], env_ids=no_extra)
                        object_.write_root_com_velocity_to_sim(state[:, 7:], env_ids=no_extra)

        # Update per-motion contact offsets, pose_range, object_pose_range
        motion_ids_for_envs = self.motion_ids[env_ids]
        if hasattr(self, "_contact_target_pos_offset_per_motion"):
            self.contact_target_pos_offset[env_ids] = self._contact_target_pos_offset_per_motion[motion_ids_for_envs]
        if hasattr(self, "_contact_eef_pos_offset_per_motion"):
            self.contact_eef_pos_offset[env_ids] = self._contact_eef_pos_offset_per_motion[motion_ids_for_envs]
        if hasattr(self, "_pose_range_per_motion"):
            self.pose_range = self._pose_range_per_motion[motion_ids_for_envs[0]]  # scalar per-motion (same for all envs in batch)
        if hasattr(self, "_object_pose_range_per_motion"):
            self.object_pose_range = self._object_pose_range_per_motion[motion_ids_for_envs[0]]

    def sample_init(self, env_ids):
        super().sample_init(env_ids)

        init_object_pos = self._motion_reset.body_pos_w[:, self.object_body_id_motion]
        init_object_quat = self._motion_reset.body_quat_w[:, self.object_body_id_motion]

        rand_samples = self.env.random_uniform(
            self.object_pose_range[:, 0],
            self.object_pose_range[:, 1],
            (len(env_ids), 6),
            env_ids=env_ids,
        )

        init_object_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        init_object_quat = quat_mul(init_object_quat, orientations_delta)
        
        init_object_state_w = self.object.data.default_root_state[env_ids]
        init_object_state_w[:, 0:3] = init_object_pos + self.env.scene.env_origins[env_ids]
        init_object_state_w[:, 3:7] = init_object_quat
        init_object_state_w[:, 7:]  = 0.0  # zero velocity

        self.object.write_root_link_pose_to_sim(init_object_state_w[:, 0:7], env_ids=env_ids)
        self.object.write_root_com_velocity_to_sim(init_object_state_w[:, 7:], env_ids=env_ids)

        if self.object2 is not None:
            init_object2_pos = self._motion_reset.body_pos_w[:, self.object2_body_id_motion]
            init_object2_quat = self._motion_reset.body_quat_w[:, self.object2_body_id_motion]
            rand_samples = self.env.random_uniform(
                self.object2_pose_range[:, 0],
                self.object2_pose_range[:, 1],
                (len(env_ids), 6),
                env_ids=env_ids,
            )
            init_object2_pos += rand_samples[:, 0:3]
            orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
            init_object2_quat = quat_mul(init_object2_quat, orientations_delta)
            init_object2_state_w = self.object2.data.default_root_state[env_ids]
            init_object2_state_w[:, 0:3] = init_object2_pos + self.env.scene.env_origins[env_ids]
            init_object2_state_w[:, 3:7] = init_object2_quat
            init_object2_state_w[:, 7:]  = 0.0  # zero velocity

            self.object2.write_root_link_pose_to_sim(init_object2_state_w[:, 0:7], env_ids=env_ids)
            self.object2.write_root_com_velocity_to_sim(init_object2_state_w[:, 7:], env_ids=env_ids)

        for object_, object_body_id_motion in zip(self.extra_objects, self.extra_object_body_id_motion):
            init_object_pos = self._motion_reset.body_pos_w[:, object_body_id_motion]
            init_object_quat = self._motion_reset.body_quat_w[:, object_body_id_motion]

            init_object_pos += rand_samples[:, 0:3]
            init_object_quat = quat_mul(init_object_quat, orientations_delta)

            init_object_state_w = object_.data.default_root_state[env_ids]
            init_object_state_w[:, 0:3] = init_object_pos + self.env.scene.env_origins[env_ids]
            init_object_state_w[:, 3:7] = init_object_quat
            init_object_state_w[:, 7:]  = 0.0  # zero velocity

            # For envs whose motion has no extra object (e.g. box_1), park the object far away
            # to avoid spurious collisions at the origin.
            if hasattr(self, "has_extra_object_mask"):
                no_extra = ~self.has_extra_object_mask[env_ids]
                if no_extra.any():
                    park_pos = self.env.scene.env_origins[env_ids[no_extra]].clone()
                    park_pos[:, 2] -= 1000.0
                    init_object_state_w[no_extra, 0:3] = park_pos

            object_.write_root_link_pose_to_sim(init_object_state_w[:, 0:7], env_ids=env_ids)
            object_.write_root_com_velocity_to_sim(init_object_state_w[:, 7:], env_ids=env_ids)

        # robot_pos_w = self.asset.data.root_link_pos_w[env_ids]
        # robot_quat_w = self.asset.data.root_link_quat_w[env_ids]
        # object_pos_b = quat_apply_inverse(robot_quat_w, (init_object_pos + self.env.scene.env_origins[env_ids]) - robot_pos_w)
        # from isaaclab.utils.math import quat_conjugate
        # object_quat_b = quat_mul(quat_conjugate(robot_quat_w), init_object_quat)
        # print(f"Object initial position in robot frame: {object_pos_b}, orientation: {object_quat_b}")

        if self.object_joint_idx_asset is not None:
            init_joint_pos = self._motion_reset.joint_pos[:, self.object_joint_idx_motion]
            init_joint_vel = self._motion_reset.joint_vel[:, self.object_joint_idx_motion]

            joint_pos_noise = self.env.random_uniform(
                -1,
                1,
                (init_joint_pos.shape[0], init_joint_pos.shape[1]),
                env_ids=env_ids,
            ) * self.object_init_joint_pos_noise
            joint_vel_noise = self.env.random_uniform(
                -1,
                1,
                (init_joint_vel.shape[0], init_joint_vel.shape[1]),
                env_ids=env_ids,
            ) * self.object_init_joint_vel_noise
            
            init_joint_pos += joint_pos_noise
            init_joint_vel += joint_vel_noise
            
            joint_pos_limits = self.object.data.soft_joint_pos_limits[env_ids]
            joint_vel_limits = self.object.data.soft_joint_vel_limits[env_ids]
            init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
            init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

            self.object.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=env_ids, joint_ids=self.object_joint_idx_asset)

        if self.object2 is not None and self.object2_joint_idx_asset is not None:
            init_joint_pos = self._motion_reset.joint_pos[:, self.object2_joint_idx_motion]
            init_joint_vel = self._motion_reset.joint_vel[:, self.object2_joint_idx_motion]

            joint_pos_noise = self.env.random_uniform(
                -1,
                1,
                (init_joint_pos.shape[0], init_joint_pos.shape[1]),
                env_ids=env_ids,
            ) * self.object_init_joint_pos_noise
            joint_vel_noise = self.env.random_uniform(
                -1,
                1,
                (init_joint_vel.shape[0], init_joint_vel.shape[1]),
                env_ids=env_ids,
            ) * self.object_init_joint_vel_noise
            
            init_joint_pos += joint_pos_noise
            init_joint_vel += joint_vel_noise
            
            joint_pos_limits = self.object2.data.soft_joint_pos_limits[env_ids]
            joint_vel_limits = self.object2.data.soft_joint_vel_limits[env_ids]
            init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
            init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

            self.object2.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=env_ids, joint_ids=self.object2_joint_idx_asset)

    def _save_motion(self):
        motion_data: TensorDict = torch.cat(self.motion_frames, dim=0)
        motion_data = motion_data[25:].numpy()
        motion_data["object_contact"] = self._object_contact[25:].cpu().numpy()
        motion_data['body_contact'] = self.body_contact[25:].cpu().numpy()
        motion_data['feet_contact'] = self._feet_contact[25:].cpu().numpy()
        object_asset_names = [self.object_asset_name]
        if self.object2_asset_name is not None:
            object_asset_names.append(self.object2_asset_name)
        motion_meta = {
            "joint_names": self.asset.joint_names,
            "body_names": self.asset.body_names + object_asset_names,
            "fps": int(1/self.env.step_dt),
        }
        save_dir = "record_motion"
        motion_data_path = f"{save_dir}/motion.npz"
        motion_meta_path = f"{save_dir}/meta.json"
        import os, json
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(motion_data_path, **motion_data)
        with open(motion_meta_path, "w") as f:
            json.dump(motion_meta, f, indent=4)
        print(f"Saved recorded motion to {motion_data_path} and {motion_meta_path}")
        breakpoint()

    def update(self):
        super().update()
        if hasattr(self, "motion_frames"):
            motion_frame = self.motion_frames[-1]
            # add object data to the motion frame
            object_pos_w = self.object.data.body_link_pos_w[:, self.object_body_id_asset].cpu()
            object_quat_w = self.object.data.body_link_quat_w[:, self.object_body_id_asset].cpu()
            object_lin_vel_w = self.object.data.body_com_lin_vel_w[:, self.object_body_id_asset].cpu()
            object_ang_vel_w = self.object.data.body_com_ang_vel_w[:, self.object_body_id_asset].cpu()
            if self.object2 is not None:
                object_pos_w = torch.cat([object_pos_w, self.object2.data.body_link_pos_w[:, self.object2_body_id_asset].cpu()], dim=1)
                object_quat_w = torch.cat([object_quat_w, self.object2.data.body_link_quat_w[:, self.object2_body_id_asset].cpu()], dim=1)
                object_lin_vel_w = torch.cat([object_lin_vel_w, self.object2.data.body_com_lin_vel_w[:, self.object2_body_id_asset].cpu()], dim=1)
                object_ang_vel_w = torch.cat([object_ang_vel_w, self.object2.data.body_com_ang_vel_w[:, self.object2_body_id_asset].cpu()], dim=1)
            motion_frame["body_pos_w"] = torch.cat([motion_frame["body_pos_w"], object_pos_w.unsqueeze(1)], dim=1)
            motion_frame["body_quat_w"] = torch.cat([motion_frame["body_quat_w"], object_quat_w.unsqueeze(1)], dim=1)
            motion_frame["body_lin_vel_w"] = torch.cat([motion_frame["body_lin_vel_w"], object_lin_vel_w.unsqueeze(1)], dim=1)
            motion_frame["body_ang_vel_w"] = torch.cat([motion_frame["body_ang_vel_w"], object_ang_vel_w.unsqueeze(1)], dim=1)

        self.ref_object_pos_future_w = self.future_ref_motion.body_pos_w[..., self.object_body_id_motion, :] + self.env.scene.env_origins[:, None, :]
        self.ref_object_quat_future_w = self.future_ref_motion.body_quat_w[..., self.object_body_id_motion, :]
        self.ref_object_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[..., self.object_body_id_motion, :]
        self.ref_object_pos_w = self.ref_object_pos_future_w[:, 0]
        self.ref_object_quat_w = self.ref_object_quat_future_w[:, 0]
        self.ref_object_lin_vel_w = self.ref_object_lin_vel_future_w[:, 0]
        self.object_pos_w = self.object.data.root_link_pos_w    # [num_envs, 3] 
        self.object_quat_w = self.object.data.root_link_quat_w
        self.object_lin_vel_w = self.object.data.root_link_vel_w[:,:3]

        if self.object_joint_idx_asset is not None:
            self.ref_object_joint_pos_future = self.future_ref_motion.joint_pos[..., self.object_joint_idx_motion]
            self.ref_object_joint_vel_future = self.future_ref_motion.joint_vel[..., self.object_joint_idx_motion]
            self.ref_object_joint_pos = self.ref_object_joint_pos_future[:, 0]
            self.ref_object_joint_vel = self.ref_object_joint_vel_future[:, 0]
            self.object_joint_pos = self.object.data.joint_pos[:, self.object_joint_idx_asset]
            self.object_joint_vel = self.object.data.joint_vel[:, self.object_joint_idx_asset]

        if self.object2 is not None:
            self.ref_object2_pos_future_w = self.future_ref_motion.body_pos_w[..., self.object2_body_id_motion, :] + self.env.scene.env_origins[:, None, :]
            self.ref_object2_quat_future_w = self.future_ref_motion.body_quat_w[..., self.object2_body_id_motion, :]
            self.ref_object2_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[..., self.object2_body_id_motion, :]
            self.ref_object2_pos_w = self.ref_object2_pos_future_w[:, 0]
            self.ref_object2_quat_w = self.ref_object2_quat_future_w[:, 0]
            self.ref_object2_lin_vel_w = self.ref_object2_lin_vel_future_w[:, 0]
            self.object2_pos_w = self.object2.data.root_link_pos_w    # [num_envs, 3] 
            self.object2_quat_w = self.object2.data.root_link_quat_w
            self.object2_lin_vel_w = self.object2.data.root_link_vel_w[:,:3]
            if self.object2_joint_idx_asset is not None:
                self.ref_object2_joint_pos_future = self.future_ref_motion.joint_pos[..., self.object2_joint_idx_motion]
                self.ref_object2_joint_vel_future = self.future_ref_motion.joint_vel[..., self.object2_joint_idx_motion]
                self.ref_object2_joint_pos = self.ref_object2_joint_pos_future[:, 0]
                self.ref_object2_joint_vel = self.ref_object2_joint_vel_future[:, 0]
                self.object2_joint_pos = self.object2.data.joint_pos[:, self.object2_joint_idx_asset]
                self.object2_joint_vel = self.object2.data.joint_vel[:, self.object2_joint_idx_asset]

        idx = (self.motion_starts + self.t).unsqueeze(1) + self.future_steps.unsqueeze(0)
        idx.clamp_max_(self.motion_ends.unsqueeze(1) - 1)
        self.ref_object_contact_future = self._object_contact[idx]
        self.ref_body_contact_future = self._body_contact[idx]
        self.ref_feet_contact_future = self._feet_contact[idx]
        self.ref_object_contact = self.ref_object_contact_future[:, 0]
        self.ref_body_contact = self.ref_body_contact_future[:, 0]
        self.ref_feet_contact = self.ref_feet_contact_future[:, 0]
        if self.object2 is not None:
            self.ref_object2_contact_future = self._object2_contact[idx]
            self.ref_object2_contact = self.ref_object2_contact_future[:, 0]
            self.ref_body2_contact_future = self._body2_contact[idx]
            self.ref_body2_contact = self.ref_body2_contact_future[:, 0]
        
        # contact target and eef pos
        object_pos_w = self.object.data.body_link_pos_w[:, self.object_body_id_asset]
        object_quat_w = self.object.data.body_link_quat_w[:, self.object_body_id_asset]
        self.contact_target_pos_w[:] = object_pos_w.unsqueeze(1) + quat_apply(object_quat_w.unsqueeze(1), self.contact_target_pos_offset)
        
        eef_pos_w = self.asset.data.body_link_pos_w[:, self.contact_eef_body_indices_asset]
        eef_quat_w = self.asset.data.body_link_quat_w[:, self.contact_eef_body_indices_asset]
        self.contact_eef_pos_w[:] = eef_pos_w + quat_apply(eef_quat_w, self.contact_eef_pos_offset)
        
        self.eef_contact_forces_w.zero_()
        for eef_idx, (eef_sensors, eef_sensor_indices) in enumerate(zip(self.eef_filtered_sensor, self.eef_filtered_sensor_indices)):
            for eef_sensor, eef_sensor_id in zip(eef_sensors, eef_sensor_indices):
                self.eef_contact_forces_w[:, eef_idx] += eef_sensor.data.force_matrix_w[:, eef_sensor_id, 0]

        self.eef_contact_forces_b[:] = quat_apply_inverse(object_quat_w.unsqueeze(1), self.eef_contact_forces_w)

        # contact2 target and eef pos
        if self.object2 is not None:
            object2_pos_w = self.object2.data.body_link_pos_w[:, self.object2_body_id_asset]
            object2_quat_w = self.object2.data.body_link_quat_w[:, self.object2_body_id_asset]
            self.contact2_target_pos_w[:] = object2_pos_w.unsqueeze(1) + quat_apply(object2_quat_w.unsqueeze(1), self.contact2_target_pos_offset)
            eef2_pos_w = self.asset.data.body_link_pos_w[:, self.contact2_eef_body_indices_asset]
            eef2_quat_w = self.asset.data.body_link_quat_w[:, self.contact2_eef_body_indices_asset]
            self.contact2_eef_pos_w[:] = eef2_pos_w + quat_apply(eef2_quat_w, self.contact2_eef_pos_offset)
            self.eef2_contact_forces_w.zero_()
            for eef_idx, (eef_sensors, eef_sensor_indices) in enumerate(zip(self.eef2_filtered_sensor, self.eef2_filtered_sensor_indices)):
                for eef_sensor, eef_sensor_id in zip(eef_sensors, eef_sensor_indices):
                    self.eef2_contact_forces_w[:, eef_idx] += eef_sensor.data.force_matrix_w[:, eef_sensor_id, 0]
            self.eef2_contact_forces_b[:] = quat_apply_inverse(object2_quat_w.unsqueeze(1), self.eef2_contact_forces_w)

        # Every step: re-park stairslopes for box_1 envs so PhysX kinematic targets stay fixed
        if len(self.extra_objects) > 0 and hasattr(self, "_motion_has_extra_object"):
            no_extra = (~self.has_extra_object_mask).nonzero(as_tuple=False).squeeze(-1)
            if len(no_extra) > 0:
                park_pos = self.env.scene.env_origins[no_extra].clone()
                park_pos[:, 2] -= 1000.0
                identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(len(no_extra), -1)
                park_pose = torch.cat([park_pos, identity_quat], dim=-1)
                zero_vel = torch.zeros(len(no_extra), 6, device=self.device)
                for object_ in self.extra_objects:
                    object_.write_root_link_pose_to_sim(park_pose, env_ids=no_extra)
                    object_.write_root_com_velocity_to_sim(zero_vel, env_ids=no_extra)

    def _init_debug_draw(self):
        super()._init_debug_draw()

        if self.env.backend != "isaac":
            return
        
        from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
        import isaaclab.sim as sim_utils
        vis_markers_cfg = VisualizationMarkersCfg(
            prim_path=f"/World/EefContact",
            markers={
                "left": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.3),
                        metallic=1.0,
                    )
                ),
                "right": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.3, 1.0),
                        metallic=1.0,
                    )
                ),
            }
        )
        self.eef_contact_markers = VisualizationMarkers(vis_markers_cfg)
        self.eef_contact_markers_indices = [0, 1] * (self.num_envs * self.num_eefs)
        self.eef_contact_markers_pos_w = torch.zeros(self.num_envs, 2, self.num_eefs, 3)
        if self.object2 is not None:
            self.eef2_contact_markers = VisualizationMarkers(vis_markers_cfg)
            self.eef2_contact_markers_indices = [0, 1] * (self.num_envs * self.num_eefs2)
            self.eef2_contact_markers_pos_w = torch.zeros(self.num_envs, 2, self.num_eefs2, 3)

    def debug_draw(self):
        super().debug_draw()

        if self.env.backend != "isaac":
            return
        
        self.eef_contact_markers_pos_w[:, 0, :, :] = self.contact_eef_pos_w
        self.eef_contact_markers_pos_w[:, 1, :, :] = self.contact_target_pos_w
        out_of_range_mask = ~self.ref_body_contact[:, None, :, None].expand_as(self.eef_contact_markers_pos_w)
        self.eef_contact_markers_pos_w[out_of_range_mask] = -1000.0
        
        self.eef_contact_markers.visualize(
            translations=self.eef_contact_markers_pos_w.view(-1, 3),
            marker_indices=self.eef_contact_markers_indices,
        )

        # visualize contact forces
        self.env.debug_draw.vector(
            self.contact_eef_pos_w.reshape(-1, 3),
            self.eef_contact_forces_w.reshape(-1, 3) / 80,
            color=(1.0, 1.0, 1.0, 1.0),
            size=4.0,
        )

        # draw vector from robot root to contact target

        self.env.debug_draw.vector(
            self.contact_eef_pos_w.view(-1, 3),
            (self.contact_target_pos_w - self.contact_eef_pos_w).view(-1, 3),
            color=(0, 1, 0, 1),
            size=4.0,
        )

        if self.object2 is not None:
            self.eef2_contact_markers_pos_w[:, 0, :, :] = self.contact2_eef_pos_w
            self.eef2_contact_markers_pos_w[:, 1, :, :] = self.contact2_target_pos_w
            out_of_range_mask = ~self.ref_body2_contact[:, None, :, None].expand_as(self.eef2_contact_markers_pos_w)
            self.eef2_contact_markers_pos_w[out_of_range_mask] = -1000.0
            
            self.eef2_contact_markers.visualize(
                translations=self.eef2_contact_markers_pos_w.view(-1, 3),
                marker_indices=self.eef2_contact_markers_indices,
            )

            # visualize contact forces
            self.env.debug_draw.vector(
                self.contact2_eef_pos_w.reshape(-1, 3),
                self.eef2_contact_forces_w.reshape(-1, 3) / 80,
                color=(1.0, 1.0, 0.0, 1.0),
                size=4.0,
            )

            # draw vector from robot root to contact target

            self.env.debug_draw.vector(
                self.contact2_eef_pos_w.view(-1, 3),
                (self.contact2_target_pos_w - self.contact2_eef_pos_w).view(-1, 3),
                color=(0, 1, 1, 1),
                size=4.0,
            )
