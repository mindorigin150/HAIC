from active_adaptation.envs.mdp.commands.hdmi.command import RobotObjectTracking
from active_adaptation.envs.mdp.base import Randomization as BaseRandomization

import torch
from typing import Dict, Tuple, List, TYPE_CHECKING
from omegaconf import DictConfig
from isaaclab.utils.math import quat_apply_inverse, sample_uniform


RobotObjectTrackRandomization = BaseRandomization[RobotObjectTracking]

class object_body_randomization(RobotObjectTrackRandomization):
    def __init__(
        self,
        dynamic_friction_range: Tuple[float, float]=(0.6, 1.0),
        restitution_range: Tuple[float, float]=(0.0, 0.2),
        mass_range: Tuple[float, float]=(1.0, 10.0),
        wheel_mass_range: Tuple[float, float]=(1.0, 10.0),
        static_friction_range: Tuple[float, float] | None = None,
        static_dynamic_friction_ratio_range: Tuple[float, float] | None = None,
        # Optional per-wheel material overrides (shapes 1+); if None, all shapes share the same material
        wheel_dynamic_friction_range: Tuple[float, float] | None = None,
        wheel_restitution_range: Tuple[float, float] | None = None,
        wheel_static_friction_range: Tuple[float, float] | None = None,
        wheel_static_dynamic_friction_ratio_range: Tuple[float, float] | None = None,
        object2_static_friction_range: Tuple[float, float] | None = None,
        object2_static_dynamic_friction_ratio_range: Tuple[float, float] | None = None,
        object2_dynamic_friction_range: Tuple[float, float] | None = None,
        object2_restitution_range: Tuple[float, float] | None = None,
        object2_mass_range: Tuple[float, float] | None = None,
        object2_wheel_mass_range: Tuple[float, float] | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Validate that only one of static_friction_range or static_dynamic_friction_ratio_range is specified
        if static_friction_range is not None and static_dynamic_friction_ratio_range is not None:
            raise ValueError("Cannot specify both static_friction_range and static_dynamic_friction_ratio_range")
        if static_friction_range is None and static_dynamic_friction_ratio_range is None:
            raise ValueError("Must specify either static_friction_range or static_dynamic_friction_ratio_range")
        
        self.object = self.command_manager.object
        if self.command_manager.object2 is not None:
            if object2_static_friction_range is not None and object2_static_dynamic_friction_ratio_range is not None:
                raise ValueError("Cannot specify both object2_static_friction_range and object2_static_dynamic_friction_ratio_range")
            if object2_static_friction_range is None and object2_static_dynamic_friction_ratio_range is None:
                raise ValueError("Must specify either object2_static_friction_range or object2_static_dynamic_friction_ratio_range")
            self.object2 = self.command_manager.object2
            self.object2_mass_range = object2_mass_range
            self.object2_wheel_mass_range = object2_wheel_mass_range
            self.all_indices_cpu2 = torch.arange(self.object2.num_instances)
            max_shapes2 = self.object2.root_physx_view.max_shapes
            self.shape_ids2 = torch.arange(0, max_shapes2)
            self.num_buckets2 = 64
            self.dynamic_friction_buckets2 = sample_uniform(*tuple(object2_dynamic_friction_range), (self.num_buckets2,), "cpu")
            self.restitution_buckets2 = sample_uniform(*tuple(object2_restitution_range), (self.num_buckets2,), "cpu")
            if object2_static_friction_range is not None:
                self.static_friction_buckets2 = sample_uniform(*tuple(object2_static_friction_range), (self.num_buckets2,), "cpu")
            else:
                self.static_dynamic_friction_ratio_buckets2 = sample_uniform(*tuple(object2_static_dynamic_friction_ratio_range), (self.num_buckets2,), "cpu")

        self.mass_range = mass_range
        self.wheel_mass_range = wheel_mass_range

        self.all_indices_cpu = torch.arange(self.object.num_instances)

        # randomize all shapes of the object
        max_shapes = self.object.root_physx_view.max_shapes
        self.shape_ids = torch.arange(0, max_shapes)

        self.num_buckets = 64

        # Sample dynamic friction and restitution buckets
        self.dynamic_friction_buckets = sample_uniform(*tuple(dynamic_friction_range), (self.num_buckets,), "cpu")
        self.restitution_buckets = sample_uniform(*tuple(restitution_range), (self.num_buckets,), "cpu")

        # Handle static friction based on which parameter is specified
        if static_friction_range is not None:
            self.static_friction_buckets = sample_uniform(*tuple(static_friction_range), (self.num_buckets,), "cpu")
        else:
            self.static_dynamic_friction_ratio_buckets = sample_uniform(*tuple(static_dynamic_friction_ratio_range), (self.num_buckets,), "cpu")

        # Optional per-wheel material buckets (shapes 1+)
        _has_wheel_dr = (wheel_dynamic_friction_range is not None or wheel_restitution_range is not None
                         or wheel_static_friction_range is not None or wheel_static_dynamic_friction_ratio_range is not None)
        if _has_wheel_dr:
            if wheel_static_friction_range is not None and wheel_static_dynamic_friction_ratio_range is not None:
                raise ValueError("Cannot specify both wheel_static_friction_range and wheel_static_dynamic_friction_ratio_range")
            if wheel_static_friction_range is None and wheel_static_dynamic_friction_ratio_range is None:
                raise ValueError("Must specify either wheel_static_friction_range or wheel_static_dynamic_friction_ratio_range when using wheel DR")
            _wdf = wheel_dynamic_friction_range if wheel_dynamic_friction_range is not None else dynamic_friction_range
            _wre = wheel_restitution_range if wheel_restitution_range is not None else restitution_range
            self.wheel_dynamic_friction_buckets = sample_uniform(*tuple(_wdf), (self.num_buckets,), "cpu")
            self.wheel_restitution_buckets = sample_uniform(*tuple(_wre), (self.num_buckets,), "cpu")
            if wheel_static_friction_range is not None:
                self.wheel_static_friction_buckets = sample_uniform(*tuple(wheel_static_friction_range), (self.num_buckets,), "cpu")
            else:
                self.wheel_static_dynamic_friction_ratio_buckets = sample_uniform(*tuple(wheel_static_dynamic_friction_ratio_range), (self.num_buckets,), "cpu")

    def startup(self):
        masses = self.object.data.default_mass.clone()
        inertias = self.object.data.default_inertia.clone()
        new_masses = sample_uniform(*self.mass_range, masses.shape, "cpu")
        new_wheel_masses = sample_uniform(*self.wheel_mass_range, masses.shape, "cpu")
        new_masses[:, 1:] = new_wheel_masses[:, 1:]

        scale = new_masses / masses
        masses[:] *= scale
        if inertias.ndim == 2:
            inertias[:] *= scale
        elif inertias.ndim == 3:
            inertias[:] *= scale.unsqueeze(-1)
        else:
            raise ValueError(f"Invalid shape for inertias: {inertias.shape}")
        self.object.root_physx_view.set_masses(masses, self.all_indices_cpu)
        self.object.root_physx_view.set_inertias(inertias, self.all_indices_cpu)
        assert torch.allclose(self.object.root_physx_view.get_masses(), masses, atol=1e-4)
        assert torch.allclose(self.object.root_physx_view.get_inertias(), inertias, atol=1e-4)

        materials = self.object.root_physx_view.get_material_properties().clone()
        shape = (self.object.num_instances, 1)
        dynamic_friction = self.dynamic_friction_buckets[torch.randint(0, self.num_buckets, shape)]
        restitution = self.restitution_buckets[torch.randint(0, self.num_buckets, shape)]
        if hasattr(self, "static_friction_buckets"):
            static_friction = self.static_friction_buckets[torch.randint(0, self.num_buckets, shape)]
        else:
            static_friction_ratio = self.static_dynamic_friction_ratio_buckets[torch.randint(0, self.num_buckets, shape)]
            static_friction = dynamic_friction * static_friction_ratio

        if hasattr(self, "wheel_dynamic_friction_buckets"):
            # deck (shape 0) and wheels (shapes 1+) get separate materials
            materials[:, :1, 0] = static_friction
            materials[:, :1, 1] = dynamic_friction
            materials[:, :1, 2] = restitution
            wheel_dynamic_friction = self.wheel_dynamic_friction_buckets[torch.randint(0, self.num_buckets, shape)]
            wheel_restitution = self.wheel_restitution_buckets[torch.randint(0, self.num_buckets, shape)]
            if hasattr(self, "wheel_static_friction_buckets"):
                wheel_static_friction = self.wheel_static_friction_buckets[torch.randint(0, self.num_buckets, shape)]
            else:
                wheel_static_friction_ratio = self.wheel_static_dynamic_friction_ratio_buckets[torch.randint(0, self.num_buckets, shape)]
                wheel_static_friction = wheel_dynamic_friction * wheel_static_friction_ratio
            materials[:, 1:, 0] = wheel_static_friction
            materials[:, 1:, 1] = wheel_dynamic_friction
            materials[:, 1:, 2] = wheel_restitution
        else:
            materials[:, self.shape_ids, 0] = static_friction
            materials[:, self.shape_ids, 1] = dynamic_friction
            materials[:, self.shape_ids, 2] = restitution
        self.object.root_physx_view.set_material_properties(materials.flatten(), self.all_indices_cpu)
        assert torch.allclose(self.object.root_physx_view.get_material_properties(), materials, atol=1e-4)
        # Store per-env body phys params for privileged observations
        self.object._custom_body_static_friction = static_friction.squeeze(-1).to(self.device)
        self.object._custom_body_dyn_friction = dynamic_friction.squeeze(-1).to(self.device)
        self.object._custom_body_restitution  = restitution.squeeze(-1).to(self.device)
        self.object._custom_body_mass         = new_masses[:, 0].to(self.device)
        if hasattr(self, "object2"):
            masses2 = self.object2.data.default_mass.clone()
            inertias2 = self.object2.data.default_inertia.clone()
            new_masses2 = sample_uniform(*self.object2_mass_range, masses2.shape, "cpu")
            new_wheel_masses2 = sample_uniform(*self.object2_wheel_mass_range, masses2.shape, "cpu")
            new_masses2[:, 1:] = new_wheel_masses2[:, 1:]

            scale2 = new_masses2 / masses2
            masses2[:] *= scale2
            if inertias2.ndim == 2:
                inertias2[:] *= scale2
            elif inertias2.ndim == 3:
                inertias2[:] *= scale2.unsqueeze(-1)
            else:
                raise ValueError(f"Invalid shape for inertias: {inertias2.shape}")
            self.object2.root_physx_view.set_masses(masses2, self.all_indices_cpu2)
            self.object2.root_physx_view.set_inertias(inertias2, self.all_indices_cpu2)
            assert torch.allclose(self.object2.root_physx_view.get_masses(), masses2, atol=1e-4)
            assert torch.allclose(self.object2.root_physx_view.get_inertias(), inertias2, atol=1e-4)

            materials2 = self.object2.root_physx_view.get_material_properties().clone()
            shape2 = (self.object2.num_instances, 1)
            dynamic_friction2 = self.dynamic_friction_buckets2[torch.randint(0, self.num_buckets, shape2)]
            restitution2 = self.restitution_buckets2[torch.randint(0, self.num_buckets, shape2)]
            if hasattr(self, "static_friction_buckets2"):
                static_friction2 = self.static_friction_buckets2[torch.randint(0, self.num_buckets, shape2)]
            else:
                static_friction_ratio2 = self.static_dynamic_friction_ratio_buckets2[torch.randint(0, self.num_buckets, shape2)]
                static_friction2 = dynamic_friction2 * static_friction_ratio2
            materials2[:, self.shape_ids2, 0] = static_friction2
            materials2[:, self.shape_ids2, 1] = dynamic_friction2
            materials2[:, self.shape_ids2, 2] = restitution2
            self.object2.root_physx_view.set_material_properties(materials2.flatten(), self.all_indices_cpu2)
            assert torch.allclose(self.object2.root_physx_view.get_material_properties(), materials2, atol=1e-4)

class object_joint_randomization(RobotObjectTrackRandomization):
    def __init__(
        self,
        friction_range: Tuple[float, float] | None=None,
        damping_range: Tuple[float, float] | None=None,
        armature_range: Tuple[float, float] | None=None,
        friction2_range: Tuple[float, float] | None=None,
        damping2_range: Tuple[float, float] | None=None,
        armature2_range: Tuple[float, float] | None=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if TYPE_CHECKING:
            from active_adaptation.assets.objects import CustomArticulation
        self.object: CustomArticulation = self.command_manager.object
        if hasattr(self.object, 'num_joints'):
            self.joint_id_asset = [i for i in range(self.object.num_joints)]
        if self.command_manager.object2 is not None:
            self.object2: CustomArticulation = self.command_manager.object2
            if hasattr(self.object2, 'num_joints'):
                self.joint_id_asset2 = [i for i in range(self.object2.num_joints)]
        self.friction_range = friction_range
        self.damping_range = damping_range
        self.armature_range = armature_range
        self.friction2_range = friction2_range
        self.damping2_range = damping2_range
        self.armature2_range = armature2_range

    
    def startup(self):
        if self.armature_range is not None:
            joint_armature = sample_uniform(*self.armature_range, (self.object.num_instances, 1), self.device)
            self.object.write_joint_armature_to_sim(joint_armature, joint_ids=self.joint_id_asset)
        if self.armature2_range is not None:
            joint_armature2 = sample_uniform(*self.armature2_range, (self.object.num_instances, 1), self.device)
            self.object2.write_joint_armature_to_sim(joint_armature2, joint_ids=self.joint_id_asset2)

    def reset(self, env_ids: torch.Tensor):
        if self.friction_range is not None:
            joint_friction = self.env.random_uniform(
                *self.friction_range,
                (len(env_ids),),
                env_ids=env_ids,
            )
            self.object._custom_friction[env_ids] = joint_friction
        if self.damping_range is not None:
            joint_damping = self.env.random_uniform(
                *self.damping_range,
                (len(env_ids),),
                env_ids=env_ids,
            )
            self.object._custom_damping[env_ids] = joint_damping
        if self.friction2_range is not None:
            joint_friction2 = self.env.random_uniform(
                *self.friction2_range,
                (len(env_ids),),
                env_ids=env_ids,
            )
            self.object2._custom_friction[env_ids] = joint_friction2
        if self.damping2_range is not None: 
            joint_damping2 = self.env.random_uniform(
                *self.damping2_range,
                (len(env_ids),),
                env_ids=env_ids,
            )
            self.object2._custom_damping[env_ids] = joint_damping2

# class keypoint_virtual_force(RobotTrackRandomization):
#     def __init__(
#         self,
#         body_names: str | List[str] = ".*",
#         stiffness_range: Tuple[float, float]=(20.0, 30.0),
#         annealing_steps: int=500,
#         pos_tolerance: float | Dict[str, float] = 0.0,
#         vel_tolerance: float | Dict[str, float] = 0.0,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.annealing_steps = annealing_steps

#         from isaaclab.utils.string import resolve_matching_names_values, resolve_matching_names
#         tracking_body_names = self.command_manager.tracking_keypoint_names
#         self.apply_force_body_names = resolve_matching_names(body_names, tracking_body_names)[1]
#         self.apply_force_body_indices_asset = []
#         self.apply_force_body_indices_motion = []
#         for name in self.apply_force_body_names:
#             body_idx_asset = self.command_manager.asset.body_names.index(name)
#             body_idx_motion = tracking_body_names.index(name)

#             self.apply_force_body_indices_asset.append(body_idx_asset)
#             self.apply_force_body_indices_motion.append(body_idx_motion)
        
#         self.pos_tolerance = torch.zeros(len(self.apply_force_body_names), device=self.device)
#         if isinstance(pos_tolerance, float):
#             self.pos_tolerance.fill_(pos_tolerance)
#         elif isinstance(pos_tolerance, DictConfig):
#             indices, names, values = resolve_matching_names_values(dict(pos_tolerance), self.apply_force_body_names)
#             self.pos_tolerance[indices] = torch.tensor(values, device=self.device)
#         else:
#             raise ValueError(f"Invalid type for pos_tolerance: {type(pos_tolerance)}")

#         self.vel_tolerance = torch.zeros(len(self.apply_force_body_names), device=self.device)
#         if isinstance(vel_tolerance, float):
#             self.vel_tolerance.fill_(vel_tolerance)
#         elif isinstance(vel_tolerance, DictConfig):
#             indices, names, values = resolve_matching_names_values(dict(vel_tolerance), self.apply_force_body_names)
#             self.vel_tolerance[indices] = torch.tensor(values, device=self.device)
#         else:
#             raise ValueError(f"Invalid type for vel_tolerance: {type(vel_tolerance)}")

#         self.stiffness_start = sample_uniform(*stiffness_range, (self.env.num_envs, 1, 1), self.device)
#         self.stiffness = self.stiffness_start.clone()
#         self.damping = self.stiffness.sqrt() * 2
        
#         self.ref_keypoint_pos_w = self.command_manager.ref_body_pos_w[:, self.apply_force_body_indices_motion]
#         self.ref_keypoint_lin_vel_w = self.command_manager.ref_body_lin_vel_w[:, self.apply_force_body_indices_motion]
    
#     def reset(self, env_ids: torch.Tensor):
#         # do not apply force in the first {decimation} steps
#         # because update is called after step after a reset
#         self.stiffness[env_ids] = 0.0
#         self.damping[env_ids] = 0.0
    
#     def update(self):
#         self.stiffness = self.stiffness_start * max(1.0 - self.env.current_iter / self.annealing_steps, 0.0)
#         self.damping = self.stiffness.sqrt() * 2

#         self.ref_keypoint_pos_w = self.command_manager.ref_body_pos_w[:, self.apply_force_body_indices_motion]
#         self.ref_keypoint_lin_vel_w = self.command_manager.ref_body_lin_vel_w[:, self.apply_force_body_indices_motion]

#     def step(self, substep):
#         if self.env.current_iter >= self.annealing_steps:
#             return
        
#         robot_keypoint_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.apply_force_body_indices_asset]
#         robot_keypoint_lin_vel_w = self.command_manager.asset.data.body_com_lin_vel_w[:, self.apply_force_body_indices_asset]

#         # compute force in world frame
#         diff_pos_w = self.ref_keypoint_pos_w - robot_keypoint_pos_w
#         diff_lin_vel_w = self.ref_keypoint_lin_vel_w - robot_keypoint_lin_vel_w
#         diff_pos_w = diff_pos_w * (diff_pos_w.abs() > self.pos_tolerance.unsqueeze(-1))
#         diff_lin_vel_w = diff_lin_vel_w * (diff_lin_vel_w.abs() > self.vel_tolerance.unsqueeze(-1))
#         self.forces_w = forces_w = self.stiffness * diff_pos_w + self.damping * diff_lin_vel_w
#         body_quat_w = self.command_manager.asset.data.body_quat_w[:, self.apply_force_body_indices_asset]
#         forces_b = quat_apply_inverse(body_quat_w, forces_w)

#         # apply force to asset
#         ext_forces_b = self.command_manager.asset._external_force_b
#         ext_forces_b[:, self.apply_force_body_indices_asset] += forces_b
#         self.command_manager.asset.has_external_wrench = True
    
#     def debug_draw(self):
#         if self.env.backend != "isaac":
#             return

#         if self.env.current_iter >= self.annealing_steps:
#             return

#         # draw force as vectors
#         body_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.apply_force_body_indices_asset]
#         self.env.debug_draw.vector(
#             body_pos_w.reshape(-1, 3),
#             (self.forces_w / self.stiffness).reshape(-1, 3) * 2.0,
#             # orange
#             color=(1.0, 0.5, 0.0, 1.0)
#         )

class push_start(RobotObjectTrackRandomization):
    def __init__(self, env, body_names, force_range = (0.2, 0.9), push_start_idx=0, min_interval=100, decay: float=0.9):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.num_bodies = len(self.body_indices)
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        self.force_range = force_range
        self.push_start_idx = push_start_idx
        self.min_interval = min_interval
        self.decay = decay
        
        with torch.device(self.env.device):
            self.last_push = torch.zeros(self.env.num_envs, len(self.body_indices), 1)
            self.forces = torch.zeros(self.env.num_envs, len(self.body_indices), 3)
            self.torques = torch.zeros(self.env.num_envs, len(self.body_indices), 3)

    def reset(self, env_ids: torch.Tensor):
        self.forces[env_ids] = 0.
        self.last_push[env_ids] = 0.

    def step(self, substep):
        if substep == 0:
            t = self.env.episode_length_buf.view(self.env.num_envs, 1, 1)
            idx = (self.command_manager.motion_starts + self.command_manager.t).view(self.env.num_envs, 1, 1) # [num_envs, 1, 1]
            i = torch.rand(self.env.num_envs, len(self.body_indices), 1, device=self.env.device) < 0.02
            i = i & ((t - self.last_push) > self.min_interval) & (idx > self.push_start_idx)
            self.last_push = torch.where(i, t, self.last_push)

            push_forces = torch.zeros_like(self.forces)
            push_forces[:, :, 0].uniform_(*self.force_range)
            push_forces[:, :, 1].uniform_(*self.force_range)
            self.forces = torch.where(i, push_forces * self.default_mass_total, self.forces * self.decay)
        self.asset.permanent_wrench_composer.set_forces_and_torques(self.forces, self.torques, body_ids=self.body_indices)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.body_indices],
            self.forces / self.default_mass_total,
            color=(1., 0.8, .4, 1.)
        )


class push_per_motion(RobotObjectTrackRandomization):
    """Per-motion push randomization: each motion gets its own body_names and force_range.

    YAML config example (3 motions):
      push_per_motion:
        body_names_per_motion:
          - ["right_wrist_yaw_link"]          # pull_cart
          - [".*_wrist_yaw_link"]             # push_cart
          - ["left_wrist_yaw_link"]           # pull_cart_mirror
        force_range_per_motion:
          - [0.2, 0.9]
          - [0.2, 0.5]
          - [0.2, 0.9]
    """
    def __init__(
        self,
        env,
        body_names_per_motion: List[List[str]],
        force_range_per_motion: List[List[float]],
        min_interval: int = 100,
        decay: float = 0.9,
    ):
        super().__init__(env)
        self.asset = self.env.scene["robot"]
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        self.min_interval = min_interval
        self.decay = decay

        # Resolve body indices per motion
        self.body_indices_per_motion: List[List[int]] = []
        for names in body_names_per_motion:
            if isinstance(names, str):
                names = [names]
            indices, _ = self.asset.find_bodies(names)
            self.body_indices_per_motion.append(indices)

        self.force_range_per_motion = [tuple(fr) for fr in force_range_per_motion]

        # Use the union of all body indices for the force/torque buffer
        all_indices = sorted(set(idx for indices in self.body_indices_per_motion for idx in indices))
        self.all_body_indices = all_indices
        self.num_bodies = len(all_indices)
        # Map body index → position in buffer
        self._idx_to_buf = {idx: i for i, idx in enumerate(all_indices)}

        with torch.device(self.env.device):
            self.last_push = torch.zeros(self.env.num_envs, self.num_bodies, 1)
            self.forces = torch.zeros(self.env.num_envs, self.num_bodies, 3)
            self.torques = torch.zeros(self.env.num_envs, self.num_bodies, 3)

        # Per-motion buffer column masks: shape (num_motions, num_bodies)
        num_motions = len(body_names_per_motion)
        self._motion_body_mask = torch.zeros(num_motions, self.num_bodies, 1, dtype=torch.bool, device=self.env.device)
        for m, indices in enumerate(self.body_indices_per_motion):
            for idx in indices:
                self._motion_body_mask[m, self._idx_to_buf[idx], 0] = True

    def reset(self, env_ids: torch.Tensor):
        self.forces[env_ids] = 0.
        self.last_push[env_ids] = 0.

    def step(self, substep):
        if substep == 0:
            t = self.env.episode_length_buf.view(self.env.num_envs, 1, 1)
            motion_ids = self.command_manager.motion_ids  # [num_envs]

            # Build per-env body mask based on current motion
            env_body_mask = self._motion_body_mask[motion_ids]  # [N, num_bodies, 1]

            i = torch.rand(self.env.num_envs, self.num_bodies, 1, device=self.env.device) < 0.02
            i = i & ((t - self.last_push) > self.min_interval) & env_body_mask
            self.last_push = torch.where(i, t, self.last_push)

            push_forces = torch.zeros_like(self.forces)
            # Apply per-motion force range per env
            for m, force_range in enumerate(self.force_range_per_motion):
                env_mask = (motion_ids == m)  # [N]
                if env_mask.any():
                    push_forces[env_mask, :, 0].uniform_(*force_range)
                    push_forces[env_mask, :, 1].uniform_(*force_range)

            self.forces = torch.where(i, push_forces * self.default_mass_total, self.forces * self.decay)
        self.asset.permanent_wrench_composer.set_forces_and_torques(self.forces, self.torques, body_ids=self.all_body_indices)
