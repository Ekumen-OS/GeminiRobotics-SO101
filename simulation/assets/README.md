# Simulation Assets Attribution

The SO-101 robot MJCF model and STL meshes in this directory are derived from
[**TheRobotStudio/SO-ARM100**](https://github.com/TheRobotStudio/SO-ARM100) (`Simulation/SO101/`),
licensed under the [Apache License 2.0](https://github.com/TheRobotStudio/SO-ARM100/blob/main/LICENSE).
The upstream CAD model was exported with [`onshape-to-robot`](https://github.com/Rhoban/onshape-to-robot),
with STS3215 servo parameters adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini)
and [RBE501-RL-arm-project](https://github.com/Gregory119/RBE501-RL-arm-project).

## Upstream Files

- `meshes/*.stl` (13 binary STL files): copied unmodified (matching SHA-1 blobs) from
  `Simulation/SO101/assets/*.stl` in `TheRobotStudio/SO-ARM100`.
- `so101.xml` (model `so101_new_calib`): derived from `Simulation/SO101/so101_new_calib.xml`.
- `scene.xml` (model `so101_scene`): derived from `Simulation/SO101/scene.xml`.
- `centered_meshes/` (git-ignored): generated at runtime by `ui.state.ensure_centered_meshes()`
  from the compiled MuJoCo mesh buffers in `meshes/` for Three.js WebGL rendering.

## Local Modifications

### `so101.xml`
- Changed `<compiler meshdir="assets" .../>` to `meshdir="meshes"`.
- Recalibrated joint `range` and actuator `ctrlrange` limits across `shoulder_pan`, `shoulder_lift`,
  `elbow_flex`, `wrist_flex`, `wrist_roll`, and `gripper` to match physical SO-101 calibration.
- Capped the `gripper` actuator `forcerange` from `[-3.35, 3.35]` to `[-1.5, 1.5]` N·m so a grasped
  30 mm cube is held without being squeezed out during lift.
- Shifted the `gripperframe` TCP `<site>` to `pos="0.014 0 -0.103"` (pad midpoint at cube-centre height)
  and disabled collision (`contype="0" conaffinity="0"`) on the raw `wrist_roll_follower_so101_v1`
  and `moving_jaw_so101_v1` STL collision geoms.
- Added box collision pads `pad_fixed` (`contype="2"`) and `pad_moving` (`contype="4"`), plus an updated
  `moving_jaw_so101_v1` body quaternion so the pads meet parallel on a 30 mm cube.
- Added the `aruco_jaw_tag` visual box geom (`material="mat_marker_10"`, 25 mm outer tile) on the
  camera-facing (`+y`) flat face of the fixed-jaw housing (`pos="-0.0114 0.0255 -0.0244"`).
- Changed structural part material `rgba` from yellow (`1 0.82 0.12 1`) to off-white (`0.95 0.95 0.95 1`).

### `scene.xml`
- Included `so101.xml` and tuned skybox, groundplane checker colors, and directional shadow lighting.
- Added `workspace_center` body and `workspace_cam` overhead camera (`fovy="45"`).
- Added workspace boundary geoms (`bound_top`, `bound_bottom`, `bound_left`, `bound_right`) and
  four 40 mm ArUco corner marker geoms (`aruco_tl_0`, `aruco_tr_1`, `aruco_br_2`, `aruco_bl_3`)
  plus textures/materials (`mat_marker_0`..`mat_marker_3`, `mat_marker_10`).
- Added the free-joint 30 mm red target cube body (`target` / `target_geom`, `conaffinity="7"`).
