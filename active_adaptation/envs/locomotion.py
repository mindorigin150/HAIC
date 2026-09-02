import os
import json
import torch
from isaaclab.utils import configclass

import active_adaptation
from active_adaptation.envs.base import _Env

class SimpleEnv(_Env):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.robot = self.scene.articulations["robot"]

        # Register camera ray visualization callback after parent init
        if (
            self.backend == "isaac"
            and self.cfg.get("enable_cameras", False)
            and not self.cfg.enable_vla_camera
        ):
            self._debug_draw_callbacks.append(self._visualize_camera_rays)
            print("[INFO] Camera ray visualization callback registered.")
        
        if self.backend == "isaac" and self.sim.has_gui():
            from isaaclab.envs.ui import BaseEnvWindow, ViewportCameraController
            from isaaclab.envs import ViewerCfg
            # hacks to make IsaacLab happy. we don't use them.
            self.lookat_env_i = (
                self.scene._default_env_origins.cpu() 
                - torch.tensor(self.cfg.viewer.lookat)
            ).norm(dim=-1).argmin().item()
            self.cfg.viewer.env_index = self.lookat_env_i
            self.manager_visualizers = {}
            self.window = BaseEnvWindow(self, window_name="IsaacLab")
            self.viewport_camera_controller = ViewportCameraController(
                self,
                ViewerCfg(self.cfg.viewer.eye, self.cfg.viewer.lookat, origin_type="env")
            )

            look_at_env_id = self.lookat_env_i
            self.sim.set_camera_view(
                eye=self.scene.env_origins[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.eye),
                target=self.scene.env_origins[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.lookat)
            )

            # ── Isaac Sim built-in visualization panels: depth image ──
            self._vis_depth_plot = None
            try:
                import numpy as np
                import omni.ui
                h, w = self.cfg.camera_height, self.cfg.camera_width
                self._depth_byte_provider = omni.ui.ByteImageProvider()
                _placeholder = np.zeros((h, w, 4), dtype=np.uint8)
                self._depth_byte_provider.set_bytes_data(_placeholder.flatten().data, [w, h])
                with self.window.ui_window_elements["main_vstack"]:
                    omni.ui.Label("── Depth Camera (Env 0) ──",
                                  height=20, style={"color": 0xFFAAAAAA})
                    with omni.ui.Frame(width=w * 3, height=h * 3):
                        self._vis_depth_plot = omni.ui.ImageWithProvider(self._depth_byte_provider)
                print("[INFO] Depth image widget added to IsaacLab window.")
            except Exception as _e:
                print(f"[WARN] Could not create depth image widget: {_e}")
            # ──────────────────────────────────────────────────────────────

        self.action_buf: torch.Tensor = self.action_manager.action_buf
        self.last_action: torch.Tensor = self.action_manager.applied_action

    def setup_scene(self):
        import active_adaptation.envs.scene as scene

        if active_adaptation.get_backend() == "isaac":
            import isaaclab.sim as sim_utils
            from isaaclab.scene import InteractiveSceneCfg
            from isaaclab.assets import AssetBaseCfg, ArticulationCfg
            from isaaclab.sensors import ContactSensorCfg
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
            from active_adaptation.assets import ROBOTS, OBJECTS, get_asset_meta
            from active_adaptation.envs.terrain import TERRAINS
            
            env_spacing = self.cfg.viewer.get("env_spacing", 2.0)
            scene_cfg = InteractiveSceneCfg(num_envs=self.cfg.num_envs, env_spacing=env_spacing, replicate_physics=False)
            scene_cfg.sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                ),
            )
            scene_cfg.robot: ArticulationCfg = ROBOTS[self.cfg.robot.name]
            
            if hasattr(self.cfg.robot, 'override_params'):
                from active_adaptation.utils import update_class_from_dict
                update_class_from_dict(scene_cfg.robot, self.cfg.robot.override_params, _ns="")
            
            scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
            robot_type = self.cfg.robot.get("robot_type", self.cfg.robot.name)
            scene_cfg.robot.spawn.usd_path = scene_cfg.robot.spawn.usd_path.format(ROBOT_TYPE=robot_type)

            # if self.cfg.command._target_ == "active_adaptation.envs.mdp.commands.hdmi.command.RobotObjectTracking":
            if "object_asset_name" in self.cfg.command:
                extra_object_names = self.cfg.command.get("extra_object_names", [])
                for extra_obj_name in extra_object_names:
                    extra_obj_cfg = OBJECTS[extra_obj_name]
                    extra_obj_cfg.prim_path = "{ENV_REGEX_NS}/" + extra_obj_name
                    setattr(scene_cfg, extra_obj_name, extra_obj_cfg)

                obj_name = self.cfg.command.object_asset_name
                obj_contact_body_name = self.cfg.command.object_body_name

                obj_cfg = OBJECTS[obj_name]
                obj_cfg.prim_path = "{ENV_REGEX_NS}/" + obj_name
                obj_type = self.cfg.command.get("object_type", obj_name)
                obj_cfg.spawn.usd_path = obj_cfg.spawn.usd_path.format(OBJECT_TYPE=obj_type)
                print(f"Using object type {obj_type} with asset {obj_cfg.spawn.usd_path}")
                setattr(scene_cfg, obj_name, obj_cfg)

                # add contact sensor to the box
                eef_names = self.cfg.command.get("contact_eef_body_name", [])
                contact_geom_prim_path = "{ENV_REGEX_NS}/" + obj_name + "/" + obj_contact_body_name

                for eef_name in eef_names:
                    contact_sensor_name = f"{eef_name}_{obj_name}_contact_forces"
                    eef_prim_path = "{ENV_REGEX_NS}/Robot/" + eef_name
                    setattr(scene_cfg, contact_sensor_name, ContactSensorCfg(
                        prim_path=eef_prim_path,
                        history_length=0,
                        track_air_time=False,
                        filter_prim_paths_expr=[contact_geom_prim_path],
                    ))
            if "object2_asset_name" in self.cfg.command:
                obj2_name = self.cfg.command.object2_asset_name
                obj2_contact_body_name = self.cfg.command.object2_body_name

                obj2_cfg = OBJECTS[obj2_name]
                obj2_cfg.prim_path = "{ENV_REGEX_NS}/" + obj2_name
                obj2_type = self.cfg.command.get("object_type", obj2_name)
                obj2_cfg.spawn.usd_path = obj2_cfg.spawn.usd_path.format(OBJECT_TYPE=obj2_type)
                print(f"Using object type {obj2_type} with asset {obj2_cfg.spawn.usd_path}")
                setattr(scene_cfg, obj2_name, obj2_cfg)

                # add contact sensor to the box
                eef2_names = self.cfg.command.get("contact2_eef_body_name", [])
                contact2_geom_prim_path = "{ENV_REGEX_NS}/" + obj2_name + "/" + obj2_contact_body_name

                for eef_name in eef2_names:
                    contact_sensor_name = f"{eef_name}_{obj2_name}_contact_forces"
                    eef_prim_path = "{ENV_REGEX_NS}/Robot/" + eef_name
                    setattr(scene_cfg, contact_sensor_name, ContactSensorCfg(
                        prim_path=eef_prim_path,
                        history_length=0,
                        track_air_time=False,
                        filter_prim_paths_expr=[contact2_geom_prim_path],
                    ))
                    
            body_scale_rand = self.cfg.randomization.get("body_scale", None)
            if body_scale_rand is not None:
                from active_adaptation.assets.spawn import clone
                if isinstance(body_scale_rand.name, str):
                    asset = getattr(scene_cfg, body_scale_rand.name)
                    spawn_func = asset.spawn.func.__wrapped__
                    asset.spawn.func = clone(spawn_func)
                    asset.spawn.scale_range = tuple(body_scale_rand.scale_range)
                    asset.spawn.homogeneous_scale = body_scale_rand.get("homogeneous_scale", False)
                    print(f"Randomized {body_scale_rand.name} scale to {asset.spawn.scale_range}")
                else:
                    for i, name, scale_range in zip(range(len(body_scale_rand.name)), body_scale_rand.name, body_scale_rand.scale_range):
                        asset = getattr(scene_cfg, name)
                        spawn_func = asset.spawn.func.__wrapped__
                        asset.spawn.func = clone(spawn_func)
                        asset.spawn.scale_range = tuple(scale_range)
                        if hasattr(body_scale_rand, "homogeneous_scale"):
                            if isinstance(body_scale_rand.homogeneous_scale, str):
                                asset.spawn.homogeneous_scale = body_scale_rand.homogeneous_scale
                            else:
                                asset.spawn.homogeneous_scale = body_scale_rand.homogeneous_scale[i]
                        else:
                            asset.spawn.homogeneous_scale = body_scale_rand.get("homogeneous_scale", False)
                        print(f"Randomized {name} scale to {asset.spawn.scale_range}")

            scene_cfg.terrain = TERRAINS[self.cfg.terrain]
            scene_cfg.contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*(ankle_roll|wrist_.*)_link", 
                history_length=3,
                track_air_time=True
            )

            if self.cfg.get("enable_cameras", False) and not self.cfg.enable_vla_camera:
                import math
                from active_adaptation.sensors import NoisyGroupedRayCasterCameraCfg
                from isaaclab.sensors.ray_caster import patterns, MultiMeshRayCasterCfg

                # Build mesh_prim_paths: ground (static, no transform tracking) + per-env objects (dynamic)
                mesh_prim_paths = [
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr="/World/ground",
                        is_shared=True,
                        track_mesh_transforms=False,
                    ),
                    # Robot links — needed so the camera can "see" the robot's own hands and legs
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr="{ENV_REGEX_NS}/Robot/.*(hip_.*|wrist_.*|shoulder_.*)_link",
                        is_shared=False,
                        track_mesh_transforms=True,
                    ),
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr="{ENV_REGEX_NS}/Robot/.*elbow_link",
                        is_shared=False,
                        track_mesh_transforms=True,
                    ),
                ]

                # Add object if present — each env has its own moving object
                if "object_asset_name" in self.cfg.command:
                    obj_name = self.cfg.command.object_asset_name
                    mesh_prim_paths.append(
                        MultiMeshRayCasterCfg.RaycastTargetCfg(
                            prim_expr=f"{{ENV_REGEX_NS}}/{obj_name}",
                            is_shared=False,
                            track_mesh_transforms=True,
                        )
                    )

                if "extra_object_names" in self.cfg.command:
                    obj_names = self.cfg.command.extra_object_names
                    for obj_name in obj_names:
                        mesh_prim_paths.append(
                            MultiMeshRayCasterCfg.RaycastTargetCfg(
                                prim_expr=f"{{ENV_REGEX_NS}}/{obj_name}",
                                is_shared=False,
                                track_mesh_transforms=True,
                            )
                        )

                # Use NoisyGroupedRayCasterCamera to support multiple mesh targets
                camera_dr = self.cfg.get("camera_dr", {})
                ray_camera = NoisyGroupedRayCasterCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
                    mesh_prim_paths=mesh_prim_paths,  # type: ignore[arg-type]
                    offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
                        # D435 camera position relative to torso_link
                        pos=(0.04764571478 + 0.0039635 - 0.0042 * math.cos(math.radians(48)),  # 0.0487988662332928
                             0.015,
                             0.46268178553 - 0.044 + 0.0042 * math.sin(math.radians(48)) + 0.016), # 0.4378029937970051
                        rot=(math.cos(math.radians(0.5) / 2) * math.cos(math.radians(48) / 2),  # 0.9135367613482678
                             math.sin(math.radians(0.5) / 2), # 0.004363309284746571
                             math.sin(math.radians(48) / 2), # 0.4067366430758002
                             0.0),  # ~47.6 degree pitch down
                        convention="world",
                    ),
                    ray_alignment="yaw",
                    pattern_cfg=patterns.PinholeCameraPatternCfg(
                        focal_length=1.0,
                        horizontal_aperture=2*math.tan(math.radians(90.05)/2),  # 87° FOVx (D435 spec)
                        vertical_aperture=2*math.tan(math.radians(58.76)/2),    # 58° FOVy (D435 spec)
                        height=self.cfg.camera_height,
                        width=self.cfg.camera_width,
                    ),
                    data_types=["distance_to_image_plane"],
                    update_period=0.02,
                    debug_vis=self.cfg.get("debug_vis", False),
                    min_distance=0.1,
                    max_distance=4.0,
                    # extrinsics DR
                    extrinsics_pos_noise_std=tuple(camera_dr.get("extrinsics_pos_noise_std", [0.0, 0.0, 0.0])),
                    extrinsics_rot_noise_std=tuple(camera_dr.get("extrinsics_rot_noise_std", [0.0, 0.0, 0.0])),
                    # intrinsics DR
                    intrinsics_focal_length_noise_std=camera_dr.get("intrinsics_focal_length_noise_std", 0.0),
                    intrinsics_aperture_noise_std=camera_dr.get("intrinsics_aperture_noise_std", 0.0),
                )
                scene_cfg.tiled_camera = ray_camera
            elif self.cfg.enable_vla_camera:
                from active_adaptation.vla.camera import haic_vla_camera_cfg

                scene_cfg.vla_camera = haic_vla_camera_cfg()
            
            sim_cfg = sim_utils.SimulationCfg(
                dt=self.cfg.sim.isaac_physics_dt,
                render=sim_utils.RenderCfg(
                    rendering_mode=(
                        "performance" if self.cfg.enable_vla_camera else "quality"
                    ),
                    # antialiasing_mode="FXAA",
                    # enable_global_illumination=True,
                    # enable_reflections=True,
                ),
                device=f"cuda:{active_adaptation.get_local_rank()}"
            )
            
            # slightly reduces GPU memory usage
            # sim_cfg.physx.gpu_max_rigid_contact_count = 2**21
            # sim_cfg.physx.gpu_max_rigid_patch_count = 2**21
            sim_cfg.physx.gpu_found_lost_pairs_capacity = 2538320*5 # 2**20
            sim_cfg.physx.gpu_found_lost_aggregate_pairs_capacity = 61999079*4 + 2**24 # 2**26
            sim_cfg.physx.gpu_total_aggregate_pairs_capacity = 2**23*5
            # Four partitions reduce scheduling overhead for the 4096-env PullCart workload.
            sim_cfg.physx.gpu_max_num_partitions = 4
            sim_cfg.physx.enable_stabilization = False
            # sim_cfg.physx.gpu_collision_stack_size = 2**25
            # sim_cfg.physx.gpu_heap_capacity = 2**24
            
            self.sim, self.scene = scene.create_isaaclab_sim_and_scene(sim_cfg, scene_cfg)

            # set camera view for "/OmniverseKit_Persp" camera
            self.sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)
            try:
                import omni.replicator.core as rep
                # create render product
                self._render_product = rep.create.render_product(
                    "/OmniverseKit_Persp", tuple(self.cfg.viewer.resolution)
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
                # self._seg_annotator = rep.AnnotatorRegistry.get_annotator(
                #     "instance_id_segmentation_fast", 
                #     device="cpu",
                # )
                # self._seg_annotator.attach([self._render_product])
                # for _ in range(4):
                #     self.sim.render()
            except ModuleNotFoundError as e:
                print("Set app.enable_cameras=true to use cameras.")
            
            try:
                from active_adaptation.utils.debug import DebugDraw
                self.debug_draw = DebugDraw()
                print("[INFO] Debug Draw API enabled.")
            except ModuleNotFoundError:
                print()

            asset_meta = get_asset_meta(self.scene["robot"])
            path = os.path.join(os.getcwd(), "asset_meta.json")
            print(f"Saving asset meta to {path}")
            with open(path, "w") as f:
                json.dump(asset_meta, f, indent=4)
        else:
            from active_adaptation.envs.mujoco import MJScene, MJSim
            from active_adaptation.assets_mjcf import ROBOTS

            @configclass
            class SceneCfg:
                robot = ROBOTS[self.cfg.robot.name]
                contact_forces = "robot"
            
            self.scene = MJScene(SceneCfg())
            self.sim = MJSim(self.scene)

        
    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.command_manager.sample_init(env_ids)
        if init_root_state is not None and not self.robot.is_fixed_base:
            self.robot.write_root_state_to_sim(
                init_root_state, 
                env_ids=env_ids
            )
        self.stats[env_ids] = 0.

    def render(self, mode: str="human"):
        # look_at_env_id = self.lookat_env_i
        # self.sim.set_camera_view(
        #     eye=self.robot.data.root_pos_w[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.eye),
        #     target=self.robot.data.root_pos_w[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.lookat)
        # )
        return super().render(mode)

    def update_vis_ui(self, td=None):
        if self._vis_depth_plot is None:    
            return
        import numpy as np

        if self.cfg.get("enable_cameras", False):
            cam = self.scene.sensors.get("tiled_camera")
            if cam is not None and hasattr(cam, "data") and hasattr(cam.data, "output"):
                depth_output = cam.data.output.get("distance_to_image_plane", None)
                if depth_output is not None:
                    d = depth_output[0, :, :, 0].float().cpu().numpy()
                    d = np.nan_to_num(d, nan=0.0, posinf=4.0, neginf=0.0)
                    d_max = d.max()
                    d_u8 = (d / d_max * 255).astype(np.uint8) if d_max > 0 else d.astype(np.uint8)
                    h, w = d_u8.shape
                    rgba = np.dstack((d_u8, d_u8, d_u8, np.full((h, w), 255, dtype=np.uint8)))
                    try:
                        self._depth_byte_provider.set_bytes_data(rgba.flatten().data, [w, h])
                    except Exception:
                        pass

    def _visualize_camera_rays(self):
        """Visualize camera rays from the depth camera.

        This is called from the base class step() via _debug_draw_callbacks when GUI is enabled.
        """
        from isaaclab.sensors import RayCasterCamera
        cam: RayCasterCamera = self.scene.sensors.get("tiled_camera")

        if cam is not None and hasattr(cam, "data") and hasattr(cam.data, "pos_w"):
            # Visualize rays for all environments
            num_envs = cam.data.pos_w.shape[0]

            for env_id in range(num_envs):
                # Camera position in world frame
                cam_pos = cam.data.pos_w[env_id]  # [3]

                # Ray hit points in world frame
                ray_hits = cam.ray_hits_w[env_id]  # [num_rays, 3]

                # Draw lines from camera position to hit points using red color
                self.debug_draw.vector(
                    x=cam_pos.unsqueeze(0).expand(len(ray_hits), -1),
                    v=ray_hits - cam_pos,
                    size=1.0,
                    color=(1.0, 0.0, 0.0, 0.5)  # red with transparency
                )
