"""State-based expert for grasping a translating and rotating cube in ManiSkill.

The script provides three variants under one predicted contact point/time:
    predictive_intercept: predict the contact state without terminal velocity sync;
    linear_sync:          additionally synchronize planar linear velocity;
    full_sync:            additionally synchronize target yaw angular velocity.

This is the fixed-direction expert baseline. The target currently starts with a y
offset and translates along +Y. A future random-direction version should generalize
both target initialization and the y-specific velocity servo to planar XY motion.

Yaw control uses a projected local axis of the rigid ``panda_hand`` link. It does not
calibrate hand yaw from the line between the two finger link origins, because those
origins are separated mainly along Z in this ManiSkill model and their tiny XY
projection can create a false diagonal grasp orientation.

The scene uses ManiSkill's default rendering and tabletop. No custom sky background
or render-only table hiding is included.
"""

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien  # noqa: F401
import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose


# ---------- generic actor helpers ----------
def wake_cube(cube: Any) -> None:
    """Wake the cube's underlying PhysX rigid body.

    ``Actor.wake_up`` is an unimplemented stub in this mani_skill version (see
    ``PhysxRigidDynamicComponentStruct`` in ``mani_skill/utils/structs/base.py``);
    only the raw PhysX body at ``cube._bodies[0]`` actually exposes ``wake_up()``.
    """
    cube._bodies[0].wake_up()


def tune_cube_physics(cube: Any, zero_damping: bool = True, zero_sleep: bool = True) -> None:
    if zero_damping:
        cube.set_linear_damping(0.0)
        cube.set_angular_damping(0.0)
    if zero_sleep:
        # set_sleep_threshold is only implemented on the raw PhysX body, not the
        # mani_skill Actor wrapper.
        cube._bodies[0].set_sleep_threshold(0.0)
        wake_cube(cube)


# ---------- marker helpers ----------
def create_visual_marker(scene, name: str, radius: float, color_rgba):
    """Build a small kinematic sphere used only as a visual debug marker.

    ``scene`` is the env's ``ManiSkillScene`` (``env.unwrapped.scene``), which
    exposes ``create_actor_builder`` directly; ``ActorBuilder.add_sphere_visual``
    and ``build_kinematic`` accept these keyword arguments in this sapien/mani_skill
    version.
    """
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=list(color_rgba))
    builder.add_sphere_visual(radius=radius, material=material)
    return builder.build_kinematic(name=name)


def set_marker_pose(marker: Any, p_xyz) -> None:
    if marker is None:
        return
    marker.set_pose(Pose.create_from_pq(p=np.array(p_xyz, dtype=np.float32), q=[1, 0, 0, 0]))


# ---------- env ----------
@register_env("DriftPickCubeAxisYawSyncBottomRefZLock-v0", max_episode_steps=1000)
class DriftPickCubeAxisYawSyncBottomRefZLockEnv(PickCubeEnv):
    def __init__(self, *args, robot_uids="panda", cube_xyz=(0.15, 0.0, 0.15), **kwargs):
        self.cube_xyz = cube_xyz
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        b = len(env_idx)
        device = self.device
        options = options or {}

        def get_scalar(name, default=0.0):
            val = options.get(name, default)
            if isinstance(val, torch.Tensor):
                return val.to(device).reshape(-1)
            return torch.full((b,), float(val), device=device)

        y_offset = get_scalar("y_offset", 0.0)
        drift_speed = get_scalar("drift_speed", 0.0)
        spin_speed = get_scalar("spin_speed", 0.0)
        cube_z = get_scalar("cube_z", self.cube_xyz[2])

        p = torch.zeros((b, 3), device=device)
        p[..., 0] = self.cube_xyz[0]
        p[..., 1] = self.cube_xyz[1] + y_offset
        p[..., 2] = cube_z
        self.cube.set_pose(Pose.create_from_pq(p=p, q=[1, 0, 0, 0]))

        v = torch.zeros((b, 3), device=device)
        v[..., 1] = drift_speed
        self.cube.set_linear_velocity(v)

        w = torch.zeros((b, 3), device=device)
        w[..., 2] = spin_speed
        self.cube.set_angular_velocity(w)


# ---------- geometry helpers ----------
def quat_to_euler_xyz_wxyz(q):
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def quat_to_rotmat_wxyz(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float32)


def projected_local_axis_yaw_wxyz(q, axis_index: int = 0, bias_deg: float = 0.0) -> float:
    """Yaw of a rigid hand-frame axis projected into the world XY plane.

    ``axis_index=0`` selects panda_hand local +X. For the top-down Panda pose,
    local X and Y both lie in the grasp plane and differ by 90 degrees. Since the
    cube's two face-axis families also differ by 90 degrees, either hand axis can
    represent a face-aligned grasp. The key is to use the rigid hand frame instead
    of the unreliable XY projection of finger-link-origin differences.
    """
    R = quat_to_rotmat_wxyz(q)
    primary = int(axis_index) if int(axis_index) in (0, 1) else 0
    for idx in (primary, 1 - primary):
        axis_xy = np.asarray(R[:2, idx], dtype=np.float32)
        n = float(np.linalg.norm(axis_xy))
        if n > 1e-8:
            axis_xy /= n
            return wrap_to_pi(
                math.atan2(float(axis_xy[1]), float(axis_xy[0]))
                + math.radians(float(bias_deg))
            )
    return wrap_to_pi(math.radians(float(bias_deg)))

def np_pose(x) -> np.ndarray:
    """Pull env 0 out of a mani_skill batched torch tensor (pose.p/.q, velocities, ...) as numpy."""
    return x[0].cpu().numpy()


def actor_linear_velocity(actor: Any) -> np.ndarray:
    """World-frame linear velocity of a mani_skill Actor or Link.

    Both structs expose ``linear_velocity`` directly (see
    ``PhysxRigidBodyComponentStruct`` in ``mani_skill/utils/structs/base.py``).
    """
    return np_pose(actor.linear_velocity).astype(np.float32)


def actor_angular_velocity(actor: Any) -> np.ndarray:
    """World-frame angular velocity of a mani_skill Actor or Link (see above)."""
    return np_pose(actor.angular_velocity).astype(np.float32)


def wrap_to_pi(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass
class AngleTracker:
    """Unwrap a periodic angle and estimate its angular rate."""

    previous: Optional[float] = None
    unwrapped: Optional[float] = None

    def update(self, angle: float, dt: float) -> tuple[float, float]:
        angle = float(angle)
        if self.previous is None:
            self.previous = angle
            self.unwrapped = angle
            return angle, 0.0

        delta = wrap_to_pi(angle - self.previous)
        self.previous = angle
        self.unwrapped = float(self.unwrapped + delta)
        return self.unwrapped, float(delta / dt)


def low_pass(previous: Optional[float], current: float, alpha: float) -> float:
    """Apply a first-order low-pass filter; the first sample initializes it."""
    return float(current) if previous is None else float(alpha * current + (1.0 - alpha) * previous)


def find_finger_links(uw):
    fingers = []
    for lk in uw.agent.robot.get_links():
        name = lk.name.lower()
        if ("leftfinger" in name) or ("rightfinger" in name):
            fingers.append(lk)
    if len(fingers) >= 2:
        fingers = sorted(fingers, key=lambda x: x.name)
        return fingers[0], fingers[1]
    raise RuntimeError("Could not find left/right finger links")


def find_hand_like_link(uw):
    exact = []
    loose = []
    for lk in uw.agent.robot.get_links():
        name = lk.name.lower()
        if "finger" in name:
            continue
        if name == "panda_hand" or name.endswith("_hand"):
            exact.append(lk)
        elif "hand" in name or name.endswith("link8"):
            loose.append(lk)
    if exact:
        return exact[0], exact[0].name, "hand"
    if loose:
        return loose[0], loose[0].name, "hand"
    raise RuntimeError("Could not find panda_hand or hand-like robot link")


def yaw_from_front_normal(axis_open):
    """Yaw of the gripper FRONT direction, not the jaw opening axis.
    If jaw axis points left-right across the fingers, the visually 'front-facing'
    direction is the planar normal to that axis.
    """
    axis_xy = np.array([axis_open[0], axis_open[1]], dtype=np.float32)
    n = np.linalg.norm(axis_xy)
    if n < 1e-8:
        return 0.0
    axis_xy = axis_xy / n
    normal_xy = np.array([-axis_xy[1], axis_xy[0]], dtype=np.float32)
    return math.atan2(float(normal_xy[1]), float(normal_xy[0]))


def axis_yaw_err(target_yaw: float, current_yaw: float) -> float:
    """Yaw error for an AXIS (180 deg periodic), not a directed heading."""
    return 0.5 * wrap_to_pi(2.0 * (target_yaw - current_yaw))

def read_gripper_geometry(uw, axis_z_offset=0.0):
    """Read the hand control point, finger midpoint and jaw-opening axis once."""
    left_finger, right_finger = find_finger_links(uw)
    left_p = np_pose(left_finger.pose.p).astype(np.float32)
    right_p = np_pose(right_finger.pose.p).astype(np.float32)
    finger_mid = (0.5 * (left_p + right_p)).astype(np.float32)

    axis_open = right_p - left_p
    axis_norm = float(np.linalg.norm(axis_open))
    if axis_norm < 1e-8:
        raise RuntimeError("Finger links have coincident positions; yaw axis is undefined")
    axis_open = (axis_open / axis_norm).astype(np.float32)

    hand_link, source_name, _ = find_hand_like_link(uw)
    axis_point = np_pose(hand_link.pose.p).copy()
    axis_point[2] += float(axis_z_offset)
    return axis_point.astype(np.float32), source_name, finger_mid, axis_open, left_p, right_p, hand_link

@dataclass
class InterceptPlan:
    plan_step: int
    plan_time: float
    wait_ready_time: float
    hit_time: float
    lin_start_time: float
    descend_start_time: float
    close_start_time: float
    hit_pos: np.ndarray
    wait_pos: np.ndarray
    close_pos: np.ndarray
    target_vel: np.ndarray
    target_wz: float
    hit_yaw: float
    wait_yaw: float
    yaw_delta_wait: float
    tau: float
    lin_accel_time: float
    lin_total_time: float


def choose_face_target_yaw(
    front_yaw_now: float,
    cube_yaw_hit: float,
    cube_wz: float,
    face_offsets,
    rot_lead: float,
):
    """Choose the easiest equivalent cube-face AXIS at the predicted hit time.

    A gripper/cube face axis is 180-degree periodic, not a directed 360-degree
    heading.  The previous implementation searched only 2*pi equivalents.  That
    could select a needlessly long turn, trigger the max-yaw-turn clamp, and then
    move the reference away from the actual predicted cube face.

    Here every face-axis candidate is expanded with k*pi equivalents before the
    wait-yaw backsolve, so the shortest physically equivalent axis is selected.
    """
    best = None
    for idx, off in enumerate(face_offsets):
        base_axis = wrap_to_pi(float(cube_yaw_hit + off))
        for k in range(-4, 5):
            # Face/gripper axes are equivalent after a 180-degree flip.
            hit_yaw_u = float(base_axis + math.pi * k)
            wait_yaw_u = float(hit_yaw_u - cube_wz * rot_lead)
            delta_wait = float(wait_yaw_u - front_yaw_now)
            turn_pen = 0.003 * abs(k)
            cost = abs(delta_wait) + turn_pen
            if best is None or cost < best[0]:
                best = (cost, idx, off, hit_yaw_u, wait_yaw_u, delta_wait)
    assert best is not None
    _, idx, off, hit_yaw_u, wait_yaw_u, delta_wait = best
    return int(idx), float(off), float(hit_yaw_u), float(wait_yaw_u), float(delta_wait)


def linear_backtrack_distance(speed: float, accel_abs: float, total_time: float) -> tuple[float, float]:
    speed = float(abs(speed))
    accel_abs = max(1e-6, float(abs(accel_abs)))
    total_time = max(0.0, float(total_time))
    t_acc = min(total_time, speed / accel_abs if speed > 1e-8 else 0.0)
    s_acc = 0.5 * accel_abs * t_acc * t_acc
    s_const = speed * max(0.0, total_time - t_acc)
    return float(s_acc + s_const), float(t_acc)


def linear_profile_along_track(elapsed: float, speed: float, accel_abs: float, total_time: float):
    speed = float(abs(speed))
    accel_abs = max(1e-6, float(abs(accel_abs)))
    total_time = max(1e-6, float(total_time))
    elapsed = float(np.clip(elapsed, 0.0, total_time))
    t_acc = min(total_time, speed / accel_abs if speed > 1e-8 else 0.0)
    if elapsed <= t_acc:
        s = 0.5 * accel_abs * elapsed * elapsed
        v = accel_abs * elapsed
    else:
        s_acc = 0.5 * accel_abs * t_acc * t_acc
        s = s_acc + speed * (elapsed - t_acc)
        v = speed
    return float(s), float(v), float(t_acc)

def choose_intercept_plan(
    step_idx: int,
    now_t: float,
    cube_p: np.ndarray,
    cube_v: np.ndarray,
    cube_yaw: float,
    cube_wz: float,
    grasp_center: np.ndarray,
    front_yaw_unwrapped: float,
    args,
):
    """Predictive translation plan plus a z-yaw/wz reference.

    Predict a planar intercept and a z-yaw/angular-rate reference. For rotation, choose a cube face
    axis at hit time, then backsolve a wait yaw so that launch/descend/close can
    spin with cube_wz and arrive near hit_yaw at hit_time.
    """
    v_xy = cube_v[:2].astype(np.float64)
    speed = float(np.linalg.norm(v_xy))
    if speed < 1e-8:
        dir_xy = np.array([0.0, 1.0], dtype=np.float64)
    else:
        dir_xy = v_xy / speed

    tau_grid = np.arange(args.closest_tau_min, args.closest_tau_max + 0.5 * args.plan_t_step, args.plan_t_step)

    lin_total_time = max(
        float(args.launch_time),
        speed / max(1e-6, float(args.xy_accel_budget)) + float(args.lin_time_margin),
    )
    launch_backtrack_s, lin_accel_time = linear_backtrack_distance(
        speed=speed,
        accel_abs=float(args.xy_accel_budget),
        total_time=lin_total_time,
    )
    # The launch profile must exactly cover the backtracked distance by hit_time.

    best = None
    for tau in tau_grid:
        tau = float(tau)
        hit_time = now_t + tau
        hit_pos = (cube_p + cube_v * tau).astype(np.float32)
        close_pos = hit_pos.copy()
        close_pos[2] = float(hit_pos[2] + args.close_clearance)

        lin_start_time = hit_time - lin_total_time
        descend_time = max(
            float(args.descend_time),
            float(args.min_descend_time),
            abs(float(close_pos[2] - (hit_pos[2] + args.wait_hover_clearance))) / max(1e-6, float(args.z_speed_budget))
            + float(args.descend_margin),
        )
        descend_start_time = hit_time - descend_time
        close_start_time = hit_time - float(args.close_lead_time)

        wait_pos = hit_pos.copy()
        wait_pos[:2] = (hit_pos[:2].astype(np.float64) - dir_xy * launch_backtrack_s).astype(np.float32)
        wait_pos[2] = float(hit_pos[2] + args.wait_hover_clearance)

        wait_xy_dist = float(np.linalg.norm((wait_pos[:2] - grasp_center[:2]).astype(np.float64)))
        wait_z_dist = abs(float(wait_pos[2] - grasp_center[2]))
        est_wait_time = max(
            wait_xy_dist / max(1e-6, float(args.xy_speed_budget)),
            wait_z_dist / max(1e-6, float(args.z_speed_budget)),
        )
        wait_ready_time = now_t + est_wait_time + float(args.plan_safety_margin)

        # Select in XY; wait_z_dist already accounts for vertical feasibility.
        hit_dist = float(np.linalg.norm((hit_pos[:2] - grasp_center[:2]).astype(np.float64)))
        score = hit_dist + float(args.plan_tau_weight) * tau
        if best is None or score < best[0]:
            best = (
                score,
                tau,
                wait_ready_time,
                hit_time,
                lin_start_time,
                descend_start_time,
                close_start_time,
                hit_pos.astype(np.float32),
                wait_pos.astype(np.float32),
                close_pos.astype(np.float32),
                est_wait_time,
            )

    assert best is not None
    (
        score,
        tau,
        wait_ready_time,
        hit_time,
        lin_start_time,
        descend_start_time,
        close_start_time,
        hit_pos,
        wait_pos,
        close_pos,
        est_wait_time,
    ) = best

    # Backsolve wait_yaw so the rotating reference reaches hit_yaw at hit_time.
    face_offsets = [math.radians(float(args.cube_body_to_face_deg)),
                    math.radians(float(args.cube_body_to_face_deg)) + math.pi / 2.0]
    cube_yaw_hit = float(cube_yaw + cube_wz * tau)
    _, _, hit_yaw, wait_yaw, yaw_delta_wait = choose_face_target_yaw(
        front_yaw_now=float(front_yaw_unwrapped),
        cube_yaw_hit=cube_yaw_hit,
        cube_wz=float(cube_wz),
        face_offsets=face_offsets,
        rot_lead=float(hit_time - lin_start_time),
    )
    max_turn = math.radians(float(args.max_predict_yaw_turn_deg))
    if abs(float(yaw_delta_wait)) > max_turn:
        # Fallback: obey the turn budget and keep a continuous spin reference.
        yaw_delta_wait = math.copysign(max_turn, float(yaw_delta_wait))
        wait_yaw = float(front_yaw_unwrapped + yaw_delta_wait)
        hit_yaw = float(wait_yaw + float(cube_wz) * float(hit_time - lin_start_time))

    return InterceptPlan(
        plan_step=step_idx,
        plan_time=now_t,
        wait_ready_time=float(wait_ready_time),
        hit_time=float(hit_time),
        lin_start_time=float(lin_start_time),
        descend_start_time=float(descend_start_time),
        close_start_time=float(close_start_time),
        hit_pos=hit_pos,
        wait_pos=wait_pos,
        close_pos=close_pos,
        target_vel=cube_v.astype(np.float32),
        target_wz=float(cube_wz),
        hit_yaw=float(hit_yaw),
        wait_yaw=float(wait_yaw),
        yaw_delta_wait=float(yaw_delta_wait),
        tau=float(tau),
        lin_accel_time=float(lin_accel_time),
        lin_total_time=float(lin_total_time),
    )



def print_plan_debug(plan: InterceptPlan, grasp_center: np.ndarray, cube_p: np.ndarray, cube_v: np.ndarray):
    speed_xy = float(np.linalg.norm(plan.target_vel[:2]))
    wait_dist_xy = float(np.linalg.norm((plan.wait_pos[:2] - grasp_center[:2]).astype(np.float64)))
    hit_dist_xy = float(np.linalg.norm((plan.hit_pos[:2] - grasp_center[:2]).astype(np.float64)))
    print(
        "[plan] "
        f"step={plan.plan_step} plan_t={plan.plan_time:+.3f} tau={plan.tau:+.3f} hit_t={plan.hit_time:+.3f} "
        f"lin_start={plan.lin_start_time:+.3f} descend_start={plan.descend_start_time:+.3f} close_start={plan.close_start_time:+.3f} "
        f"lin_lead={plan.hit_time - plan.lin_start_time:+.3f} desc_lead={plan.hit_time - plan.descend_start_time:+.3f}"
    )
    print(
        "[plan] "
        f"cube_now=({cube_p[0]:+.3f},{cube_p[1]:+.3f},{cube_p[2]:+.3f}) "
        f"cube_v=({cube_v[0]:+.3f},{cube_v[1]:+.3f},{cube_v[2]:+.3f}) "
        f"grasp_now=({grasp_center[0]:+.3f},{grasp_center[1]:+.3f},{grasp_center[2]:+.3f}) "
        f"speed_xy={speed_xy:+.3f} wait_ready_t={plan.wait_ready_time:+.3f}"
    )
    print(
        "[plan] "
        f"hit_pos=({plan.hit_pos[0]:+.3f},{plan.hit_pos[1]:+.3f},{plan.hit_pos[2]:+.3f}) "
        f"wait_pos=({plan.wait_pos[0]:+.3f},{plan.wait_pos[1]:+.3f},{plan.wait_pos[2]:+.3f}) "
        f"close_pos=({plan.close_pos[0]:+.3f},{plan.close_pos[1]:+.3f},{plan.close_pos[2]:+.3f}) "
        f"hit_dist_xy={hit_dist_xy:+.3f} wait_dist_xy={wait_dist_xy:+.3f} "
        f"lin_total={plan.lin_total_time:+.3f} lin_acc={plan.lin_accel_time:+.3f}"
    )
    print(
        "[plan_yaw] "
        f"wait_yaw={math.degrees(plan.wait_yaw):+.1f} hit_yaw={math.degrees(plan.hit_yaw):+.1f} "
        f"delta_wait={math.degrees(plan.yaw_delta_wait):+.1f} target_wz={math.degrees(plan.target_wz):+.1f}deg/s"
    )


def print_phase_debug(
    tag: str,
    now_t: float,
    plan: InterceptPlan,
    grasp_center: np.ndarray,
    ee_v: np.ndarray,
    front_yaw: float,
    front_wz: float,
):
    pos_to_wait = (plan.wait_pos - grasp_center).astype(np.float32)
    pos_to_hit = (plan.hit_pos - grasp_center).astype(np.float32)
    pos_to_close = (plan.close_pos - grasp_center).astype(np.float32)
    print(
        f"[phase_enter] {tag} now_t={now_t:+.3f} "
        f"dt_to_hit={plan.hit_time - now_t:+.3f} "
        f"dt_to_lin={plan.lin_start_time - now_t:+.3f} "
        f"dt_to_desc={plan.descend_start_time - now_t:+.3f} "
        f"dt_to_close={plan.close_start_time - now_t:+.3f}"
    )
    print(
        f"[phase_enter] {tag} "
        f"grasp=({grasp_center[0]:+.3f},{grasp_center[1]:+.3f},{grasp_center[2]:+.3f}) "
        f"ee_v=({ee_v[0]:+.3f},{ee_v[1]:+.3f},{ee_v[2]:+.3f}) "
        f"front_yaw={math.degrees(front_yaw):+.2f} front_wz={math.degrees(front_wz):+.2f}"
    )
    print(
        f"[phase_enter] {tag} "
        f"to_wait=({pos_to_wait[0]:+.3f},{pos_to_wait[1]:+.3f},{pos_to_wait[2]:+.3f}) "
        f"to_hit=({pos_to_hit[0]:+.3f},{pos_to_hit[1]:+.3f},{pos_to_hit[2]:+.3f}) "
        f"to_close=({pos_to_close[0]:+.3f},{pos_to_close[1]:+.3f},{pos_to_close[2]:+.3f})"
    )



def control_xyz(ref_pos, ref_vel, cur_pos, cur_vel, kp_xyz, kv_xyz, ff_xyz, max_cmd=1.0):
    pos_err = (ref_pos - cur_pos).astype(np.float32)
    vel_err = (ref_vel - cur_vel).astype(np.float32)
    u = kp_xyz * pos_err + kv_xyz * vel_err + ff_xyz * ref_vel
    return np.clip(u, -max_cmd, max_cmd).astype(np.float32), pos_err, vel_err


# ---------- y-axis velocity-only controller ----------
def control_y_velocity_only(
    target_vy: float,
    cur_vy: float,
    dt: float,
    vy_int: float,
    ee_v_per_action_y: float,
    kv_action: float,
    ki_action: float,
    int_clip: float,
    action_clip: float,
):
    """Control y motion from velocity only.

    This intentionally does NOT use y position error. During dynamic intercept
    phases, the goal is to verify that the execution layer can make the gripper
    y velocity match the predicted object y velocity. Position timing can then
    be tuned separately by changing hit_time / launch_time / wait_pos.
    """
    vel_err = float(target_vy - cur_vy)
    vy_int = float(np.clip(
        vy_int + vel_err * float(dt),
        -float(int_clip),
        +float(int_clip),
    ))

    # Velocity feedforward: desired gripper y speed -> raw action_y.
    # ee_v_per_action_y is an empirical conversion factor for this env/control mode.
    action_ff = float(target_vy) / max(1e-6, float(ee_v_per_action_y))
    action_fb = float(kv_action) * vel_err
    action_i = float(ki_action) * vy_int

    action_y = action_ff + action_fb + action_i
    action_y = float(np.clip(action_y, -float(action_clip), +float(action_clip)))
    return action_y, vy_int, vel_err, action_ff, action_fb, action_i


def smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def smoothstep01_derivative(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return 6.0 * x * (1.0 - x)




def build_close_schedule():
    return np.array([+1.0, +0.7, +0.4, +0.1, -0.1, -0.25, -0.45, -0.65, -0.82, -1.0], dtype=np.float32)


def _safe_float(x):
    return float(x)


def _write_rows_csv(path_str: str, rows):
    if not path_str or not rows:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_summary_csv(path_str: str, summary: dict):
    if not path_str:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(summary.keys())
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(summary)


def _first_row(rows, predicate):
    for row in rows:
        if predicate(row):
            return row
    return None


def _max_abs(rows, key):
    vals = []
    for row in rows:
        v = float(row[key])
        if math.isfinite(v):
            vals.append(abs(v))
    return max(vals) if vals else float("nan")



def build_arg_parser():
    """All 98 knobs, grouped by which control phase or subsystem reads them.

    Grouping only reorganizes/comments this list; no default value changed and
    no flag removed (every flag here is read at least once elsewhere in the file).
    """
    ap = argparse.ArgumentParser(
        description="State-based expert for a translating and rotating cube.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- run control ----
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--debug_every", type=int, default=2)

    # ---- cube motion scenario ----
    ap.add_argument("--drift_speed", type=float, default=0.03)
    ap.add_argument("--y_offset", type=float, default=-0.60)
    ap.add_argument("--spin_speed", type=float, default=0.30)
    ap.add_argument("--cube_z", type=float, default=0.14)

    # ---- experiment / ablation / logging (do not change the control logic) ----
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method_name", type=str, default="predictive_grasp_expert")
    ap.add_argument(
        "--variant",
        choices=["predictive_intercept", "linear_sync", "full_sync"],
        default="full_sync",
        help=(
            "predictive_intercept predicts the contact point/time but reaches it with "
            "zero terminal XY velocity and zero angular-rate reference; linear_sync "
            "adds target XY velocity synchronization while holding the predicted hit "
            "yaw; full_sync additionally tracks predicted cube angular velocity."
        ),
    )
    ap.add_argument("--log_csv", type=str, default="")
    ap.add_argument("--summary_csv", type=str, default="")
    ap.add_argument("--log_every", type=int, default=1)

    # ---- cube physics (zero-gravity drift tuning) ----
    ap.add_argument("--zero_damping", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--zero_sleep", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--keep_awake_every_step", action=argparse.BooleanOptionalAction, default=True)

    # ---- gripper / cube geometry calibration ----
    ap.add_argument("--axis_z_offset", type=float, default=0.0)
    ap.add_argument("--gripper_front_bias_deg", type=float, default=-2.2)
    ap.add_argument("--cube_body_to_face_deg", type=float, default=0.0)

    # ---- intercept planning: predicted contact point/time (choose_intercept_plan) ----
    ap.add_argument("--observe_steps", type=int, default=4)
    ap.add_argument("--observe_hover_clearance", type=float, default=0.09)
    ap.add_argument("--closest_tau_min", type=float, default=0.80)
    ap.add_argument("--closest_tau_max", type=float, default=20.0)
    ap.add_argument("--wait_hover_clearance", type=float, default=0.060)
    ap.add_argument("--launch_time", type=float, default=1.80)
    ap.add_argument("--descend_time", type=float, default=1.35)
    ap.add_argument("--min_descend_time", type=float, default=1.35)
    ap.add_argument("--lin_time_margin", type=float, default=1.10)
    ap.add_argument("--descend_margin", type=float, default=0.05)
    ap.add_argument("--plan_t_step", type=float, default=0.02)
    ap.add_argument("--plan_safety_margin", type=float, default=0.05)
    ap.add_argument("--plan_tau_weight", type=float, default=0.0)
    ap.add_argument("--xy_speed_budget", type=float, default=0.35)
    ap.add_argument("--z_speed_budget", type=float, default=0.18)
    ap.add_argument("--xy_accel_budget", type=float, default=0.18)
    ap.add_argument("--close_clearance", type=float, default=0.012)
    ap.add_argument("--close_lead_time", type=float, default=0.00)

    # ---- PD gains: pre-intercept phases (observe / go_wait / wait_hold) ----
    ap.add_argument("--kp_xy_pre", type=float, default=4.0)
    ap.add_argument("--kp_z_pre", type=float, default=3.0)

    # ---- PD gains: dynamic-intercept phases (launch / descend / close base gains) ----
    ap.add_argument("--kp_xy_sync", type=float, default=3.8)
    ap.add_argument("--kp_z_sync", type=float, default=2.4)
    ap.add_argument("--kv_xy_sync", type=float, default=1.0)
    ap.add_argument("--kv_z_sync", type=float, default=0.5)
    ap.add_argument("--ff_xy_sync", type=float, default=0.45)
    ap.add_argument("--ff_z_sync", type=float, default=0.00)

    # ---- action scaling: general + go_wait phase ----
    ap.add_argument("--move_scale", type=float, default=0.24)
    # Minimum pre-positioning commands prevent a slow PD tail near wait_pos.
    ap.add_argument("--fast_go_wait", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--go_wait_move_scale", type=float, default=0.42)
    ap.add_argument("--go_wait_action_clip", type=float, default=0.42)
    ap.add_argument("--go_wait_min_xy_action", type=float, default=0.055)
    ap.add_argument("--go_wait_min_z_action", type=float, default=0.025)
    ap.add_argument("--go_wait_far_xy", type=float, default=0.025)
    ap.add_argument("--go_wait_far_z", type=float, default=0.012)
    ap.add_argument("--go_wait_xy_eps", type=float, default=0.012)
    ap.add_argument("--go_wait_z_eps", type=float, default=0.010)

    # ---- action scaling / per-axis gain overrides: launch phase ----
    ap.add_argument("--launch_y_action_scale", type=float, default=1.25)
    ap.add_argument("--launch_kp_x_scale", type=float, default=0.60)
    ap.add_argument("--launch_kp_y_scale", type=float, default=1.35)
    ap.add_argument("--launch_kv_x_scale", type=float, default=0.55)
    ap.add_argument("--launch_kv_y_scale", type=float, default=1.60)
    ap.add_argument("--launch_ff_x_scale", type=float, default=1.00)
    ap.add_argument("--launch_ff_y_scale", type=float, default=1.85)

    # ---- action scaling / per-axis gain overrides: descend phase ----
    ap.add_argument("--descend_y_action_scale", type=float, default=1.20)
    ap.add_argument("--descend_z_action_scale", type=float, default=2.35)
    ap.add_argument("--descend_z_min_action", type=float, default=0.055)
    ap.add_argument("--descend_z_far_eps", type=float, default=0.006)
    ap.add_argument("--descend_kp_x_scale", type=float, default=0.75)
    ap.add_argument("--descend_kp_y_scale", type=float, default=1.55)
    ap.add_argument("--descend_kp_z_scale", type=float, default=3.20)
    ap.add_argument("--descend_kv_x_scale", type=float, default=0.75)
    ap.add_argument("--descend_kv_y_scale", type=float, default=1.70)
    ap.add_argument("--descend_kv_z_scale", type=float, default=1.10)
    ap.add_argument("--descend_ff_x_scale", type=float, default=1.00)
    ap.add_argument("--descend_ff_y_scale", type=float, default=1.75)
    ap.add_argument("--descend_ff_z_scale", type=float, default=1.75)

    # ---- y-velocity-only controller (launch / descend / close, see control_y_velocity_only) ----
    # During launch/descend/close, replace y position tracking with pure
    # y-velocity servo. go_wait/wait_hold still use position control so the
    # gripper can reach the precomputed wait point.
    ap.add_argument("--vy_vel_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ee_v_per_action_y", type=float, default=0.75)
    ap.add_argument("--vy_kv_action", type=float, default=1.20)
    ap.add_argument("--vy_ki_action", type=float, default=0.80)
    ap.add_argument("--vy_int_clip", type=float, default=0.05)
    ap.add_argument("--vy_action_clip", type=float, default=0.14)

    # ---- yaw control ----
    ap.add_argument("--kp_yaw_sync", type=float, default=2.5)
    ap.add_argument("--kp_w_sync", type=float, default=0.15)
    ap.add_argument("--yaw_ff_gain", type=float, default=1.0)
    ap.add_argument("--a5_per_front_wz_deg", type=float, default=-1.0 / 44.8)
    ap.add_argument("--yaw_clip", type=float, default=0.45)
    ap.add_argument("--enable_yaw_sync", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--yaw_state_source",
        type=str,
        choices=["hand", "finger"],
        default="hand",
        help="Yaw state used by planning/control. 'hand' uses a projected rigid panda_hand local axis and direct world wz; "
             "'finger' is a legacy diagnostic mode and can be geometrically unreliable.",
    )
    ap.add_argument(
        "--wz_filter_alpha",
        type=float,
        default=0.35,
        help="Low-pass alpha for direct panda_hand wz used by control (1.0 disables filtering).",
    )
    ap.add_argument("--max_predict_yaw_turn_deg", type=float, default=100.0)
    ap.add_argument("--yaw_wait_wz", action=argparse.BooleanOptionalAction, default=False)

    # ---- close / post-grasp ----
    ap.add_argument("--close_ramp_step", type=int, default=3)
    ap.add_argument("--post_hold_grip", type=float, default=-0.85)
    ap.add_argument("--post_steps", type=int, default=40)
    ap.add_argument("--post_ff_xy", type=float, default=0.45)

    # ---- debug markers ----
    ap.add_argument("--show_markers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--marker_radius", type=float, default=0.006)
    ap.add_argument("--marker_z_offset", type=float, default=0.025)

    return ap

def create_env_and_reset(args):
    env = gym.make(
        "DriftPickCubeAxisYawSyncBottomRefZLock-v0",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="human" if args.render else None,
        sim_config=dict(scene_config=dict(gravity=[0.0, 0.0, 0.0])),
    )
    env.reset(
        seed=args.seed,
        options=dict(
            y_offset=args.y_offset,
            drift_speed=args.drift_speed,
            spin_speed=args.spin_speed,
            cube_z=args.cube_z,
        ),
    )
    return env


def validate_args(args):
    """Reject invalid experiment settings instead of silently repairing them."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.observe_steps < 0:
        raise ValueError("--observe_steps cannot be negative")
    if args.plan_t_step <= 0:
        raise ValueError("--plan_t_step must be positive")
    if args.closest_tau_max < args.closest_tau_min:
        raise ValueError("--closest_tau_max must be >= --closest_tau_min")
    if args.debug_every <= 0 or args.log_every <= 0:
        raise ValueError("--debug_every and --log_every must be positive")
    if args.close_ramp_step <= 0:
        raise ValueError("--close_ramp_step must be positive")
    if not 0.0 <= args.wz_filter_alpha <= 1.0:
        raise ValueError("--wz_filter_alpha must be in [0, 1]")

def resolve_method_name(args):
    if args.method_name != "predictive_grasp_expert":
        return args.method_name
    return {
        "predictive_intercept": "predictive_intercept",
        "linear_sync": "predictive_linear_sync",
        "full_sync": "predictive_full_sync",
    }[args.variant]


def get_phase_gains(phase: str, args):
    """Return xyz position/velocity/feedforward gains for a control phase."""
    if phase == "observe":
        return (
            np.array([0.0, 0.0, args.kp_z_pre], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
    if phase == "go_wait":
        return (
            np.array([args.kp_xy_pre * 1.20, args.kp_xy_pre * 1.20, args.kp_z_pre], dtype=np.float32),
            np.array([args.kv_xy_sync * 0.15, args.kv_xy_sync * 0.15, args.kv_z_sync * 0.10], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
    if phase == "wait_hold":
        return (
            np.array([args.kp_xy_pre, args.kp_xy_pre, args.kp_z_pre], dtype=np.float32),
            np.array([args.kv_xy_sync * 0.08, args.kv_xy_sync * 0.08, args.kv_z_sync * 0.08], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
    if phase == "launch":
        return (
            np.array([
                args.kp_xy_sync * args.launch_kp_x_scale,
                args.kp_xy_sync * args.launch_kp_y_scale,
                args.kp_z_sync * 0.35,
            ], dtype=np.float32),
            np.array([
                args.kv_xy_sync * args.launch_kv_x_scale,
                args.kv_xy_sync * args.launch_kv_y_scale,
                args.kv_z_sync * 0.25,
            ], dtype=np.float32),
            np.array([
                args.ff_xy_sync * args.launch_ff_x_scale,
                args.ff_xy_sync * args.launch_ff_y_scale,
                args.ff_z_sync * 0.20,
            ], dtype=np.float32),
        )
    if phase == "descend":
        return (
            np.array([
                args.kp_xy_sync * args.descend_kp_x_scale,
                args.kp_xy_sync * args.descend_kp_y_scale,
                args.kp_z_sync * args.descend_kp_z_scale,
            ], dtype=np.float32),
            np.array([
                args.kv_xy_sync * args.descend_kv_x_scale,
                args.kv_xy_sync * args.descend_kv_y_scale,
                args.kv_z_sync * args.descend_kv_z_scale,
            ], dtype=np.float32),
            np.array([
                args.ff_xy_sync * args.descend_ff_x_scale,
                args.ff_xy_sync * args.descend_ff_y_scale,
                args.ff_z_sync * args.descend_ff_z_scale,
            ], dtype=np.float32),
        )
    if phase == "close":
        return (
            np.array([args.kp_xy_sync, args.kp_xy_sync, args.kp_z_sync], dtype=np.float32),
            np.array([args.kv_xy_sync, args.kv_xy_sync, args.kv_z_sync], dtype=np.float32),
            np.array([args.ff_xy_sync, args.ff_xy_sync, 0.0], dtype=np.float32),
        )
    if phase == "post_grasp":
        return (
            np.array([args.kp_xy_sync, args.kp_xy_sync, args.kp_z_sync], dtype=np.float32),
            np.array([args.kv_xy_sync, args.kv_xy_sync, args.kv_z_sync], dtype=np.float32),
            np.array([args.post_ff_xy, args.post_ff_xy, 0.0], dtype=np.float32),
        )
    raise ValueError(f"No controller gains defined for phase: {phase}")


def get_axis_scale(phase: str, args):
    """Return action scaling for the current control phase."""
    if phase == "go_wait" and args.fast_go_wait:
        scale = np.array([args.go_wait_move_scale, args.go_wait_move_scale, args.go_wait_move_scale], dtype=np.float32)
    else:
        scale = np.array([args.move_scale, args.move_scale, args.move_scale], dtype=np.float32)

    if phase == "launch":
        scale[1] *= args.launch_y_action_scale
    elif phase == "descend":
        scale[1] *= args.descend_y_action_scale
        scale[2] *= args.descend_z_action_scale
    return scale


def compute_launch_descend_reference(plan: InterceptPlan, now_t: float, args):
    """Compute the reference pose/velocity during the dynamic intercept window."""
    launch_elapsed = float(np.clip(now_t - plan.lin_start_time, 0.0, plan.lin_total_time))
    ref_pos = plan.wait_pos.copy()
    ref_vel = np.zeros(3, dtype=np.float32)

    if args.variant == "predictive_intercept":
        # This ablation reaches the predicted hit point with a smooth position
        # profile, but intentionally ends with zero XY velocity.
        intercept_T = max(1e-6, float(plan.hit_time - plan.lin_start_time))
        intercept_tau = float(np.clip(now_t - plan.lin_start_time, 0.0, intercept_T))
        intercept_alpha = intercept_tau / intercept_T
        s_xy = smoothstep01(intercept_alpha)
        ds_xy = smoothstep01_derivative(intercept_alpha) / intercept_T
        delta_xy = (plan.hit_pos[:2] - plan.wait_pos[:2]).astype(np.float32)
        ref_pos[:2] = plan.wait_pos[:2] + float(s_xy) * delta_xy
        ref_vel[:2] = float(ds_xy) * delta_xy
        launch_s = float(intercept_alpha)
    else:
        # Linear/full sync use the same tuned 1D profile along the target's
        # planar motion direction, so terminal XY speed matches the target.
        speed_xy = float(np.linalg.norm(plan.target_vel[:2]))
        s_prog, v_prog, _ = linear_profile_along_track(
            elapsed=launch_elapsed,
            speed=speed_xy,
            accel_abs=args.xy_accel_budget,
            total_time=plan.lin_total_time,
        )
        if speed_xy < 1e-8:
            dir_xy = np.array([0.0, 1.0], dtype=np.float32)
        else:
            dir_xy = (plan.target_vel[:2] / speed_xy).astype(np.float32)
        ref_pos[:2] = plan.wait_pos[:2] + dir_xy * float(s_prog)
        ref_vel[:2] = dir_xy * float(v_prog)
        launch_s = float(np.clip(launch_elapsed / max(1e-6, plan.lin_total_time), 0.0, 1.0))

    if now_t < plan.descend_start_time:
        ref_pos[2] = float(plan.wait_pos[2])
        ref_vel[2] = 0.0
    else:
        descend_T = max(1e-6, float(plan.hit_time - plan.descend_start_time))
        descend_tau = float(np.clip(now_t - plan.descend_start_time, 0.0, descend_T))
        descend_alpha = descend_tau / descend_T
        s_desc = smoothstep01(descend_alpha)
        ds_desc = smoothstep01_derivative(descend_alpha) / descend_T
        z_delta = float(plan.close_pos[2] - plan.wait_pos[2])
        ref_pos[2] = float(plan.wait_pos[2] + s_desc * z_delta)
        ref_vel[2] = float(ds_desc * z_delta)

    return ref_pos, ref_vel, launch_s


def compute_yaw_reference(phase: str, plan: Optional[InterceptPlan], now_t: float, front_yaw: float, post_hold_yaw, args):
    """Return yaw and yaw-rate references for the current ablation variant."""
    ref_yaw = float(front_yaw)
    ref_wz = 0.0
    if not args.enable_yaw_sync or plan is None:
        return ref_yaw, ref_wz

    if args.variant in ("predictive_intercept", "linear_sync"):
        if phase in ("go_wait", "wait_hold", "launch", "descend", "close"):
            return float(plan.hit_yaw), 0.0
        if phase == "post_grasp" and post_hold_yaw is not None:
            return float(post_hold_yaw), 0.0
        return ref_yaw, ref_wz

    if phase in ("go_wait", "wait_hold"):
        ref_wz = float(plan.target_wz) if args.yaw_wait_wz else 0.0
        return float(plan.wait_yaw), ref_wz
    if phase in ("launch", "descend", "close"):
        ref_yaw = float(plan.hit_yaw + plan.target_wz * (now_t - plan.hit_time))
        return ref_yaw, float(plan.target_wz)
    if phase == "post_grasp" and post_hold_yaw is not None:
        return float(post_hold_yaw), 0.0
    return ref_yaw, ref_wz


def _vec3_fields(prefix: str, vec):
    """Create x/y/z CSV fields for a 3D vector."""
    return {
        f"{prefix}_x": _safe_float(vec[0]),
        f"{prefix}_y": _safe_float(vec[1]),
        f"{prefix}_z": _safe_float(vec[2]),
    }


def _rad_and_deg_fields(name: str, value_rad: float):
    """Log both radians and degrees so analysis scripts can use either unit."""
    return {
        name: _safe_float(value_rad),
        f"{name}_deg": _safe_float(math.degrees(value_rad)),
    }


def build_step_log_row(
    args,
    *,
    t: int,
    now_t: float,
    phase: str,
    cube_p,
    cube_v,
    cube_yaw: float,
    cube_wz: float,
    grasp_center,
    ee_v,
    front_yaw: float,
    front_yaw_unwrapped: float,
    front_wz: float,
    control_yaw_source: str,
    use_hand_state: bool,
    finger_axis_yaw: float,
    finger_axis_wz: float,
    hand_roll: float,
    hand_pitch: float,
    hand_yaw: float,
    hand_yaw_unwrapped: float,
    hand_yaw_wz_fd: float,
    hand_w_direct,
    hand_w_source: str,
    hand_link_name_diag,
    ref_pos,
    ref_vel,
    ref_yaw: float,
    ref_wz: float,
    pos_err_vec,
    vel_err_vec,
    yaw_err: float,
    w_err: float,
    action,
    plan_tau: float,
    plan_hit: float,
    time_to_hit,
    launch_s: float,
    close_start_step,
    post_start_step,
    grasp_latched: bool,
    vy_only_active: bool,
    vy_err_only: float,
):
    """Build one CSV row without mixing logging details into the controller loop."""
    rel_pos = (grasp_center - cube_p).astype(np.float32)
    rel_vel = (ee_v - cube_v).astype(np.float32)

    row = {
        "method": args.method_name,
        "variant": args.variant,
        "seed": int(args.seed),
        "t": int(t),
        "time": _safe_float(now_t),
        "phase": phase,
        "drift_speed": _safe_float(args.drift_speed),
        "spin_speed": _safe_float(args.spin_speed),
        "y_offset": _safe_float(args.y_offset),
        "control_yaw_source": str(control_yaw_source),
        "hand_axis_definition": "panda_hand_local_x_projected" if use_hand_state else "legacy_finger_projection",
        "hand_w_source": str(hand_w_source),
        "hand_link_name": str(hand_link_name_diag),
        "rel_speed": _safe_float(np.linalg.norm(rel_vel)),
        "pos_err_norm": _safe_float(np.linalg.norm(pos_err_vec)),
        "vel_err_norm": _safe_float(np.linalg.norm(vel_err_vec)),
        "action_x": _safe_float(action[0]),
        "action_y": _safe_float(action[1]),
        "action_z": _safe_float(action[2]),
        "action_yaw": _safe_float(action[5]),
        "grip_cmd": _safe_float(action[-1]),
        "plan_tau": _safe_float(plan_tau),
        "plan_hit_time": _safe_float(plan_hit),
        "time_to_hit": _safe_float(time_to_hit if time_to_hit is not None else float("nan")),
        "launch_s": _safe_float(launch_s),
        "close_start_step": int(close_start_step) if close_start_step is not None else -1,
        "post_start_step": int(post_start_step) if post_start_step is not None else -1,
        "grasp_latched": int(bool(grasp_latched)),
        "vy_only_active": int(bool(vy_only_active)),
        "vy_err_only": _safe_float(vy_err_only),
    }

    row.update(_vec3_fields("cube", cube_p))
    row.update(_vec3_fields("cube_v", cube_v))
    row.update(_rad_and_deg_fields("cube_yaw", cube_yaw))
    row.update(_rad_and_deg_fields("cube_wz", cube_wz))
    row.update(_vec3_fields("grasp", grasp_center))
    row.update(_vec3_fields("ee_v", ee_v))
    row.update(_rad_and_deg_fields("front_yaw", front_yaw))
    row["front_yaw_unwrapped"] = _safe_float(front_yaw_unwrapped)
    row.update(_rad_and_deg_fields("front_wz", front_wz))
    row.update(_rad_and_deg_fields("control_front_yaw", front_yaw))
    row.update(_rad_and_deg_fields("control_front_wz", front_wz))
    row.update(_rad_and_deg_fields("finger_axis_yaw", finger_axis_yaw))
    row.update(_rad_and_deg_fields("finger_axis_wz_fd", finger_axis_wz))
    row.update(_rad_and_deg_fields("hand_roll", hand_roll))
    row.update(_rad_and_deg_fields("hand_pitch", hand_pitch))
    row.update(_rad_and_deg_fields("hand_yaw", hand_yaw))
    row["hand_yaw_unwrapped"] = _safe_float(hand_yaw_unwrapped)
    row.update(_rad_and_deg_fields("hand_yaw_wz_fd", hand_yaw_wz_fd))
    row.update(_vec3_fields("hand_w_direct", hand_w_direct))
    row["hand_wx_direct_deg"] = _safe_float(math.degrees(hand_w_direct[0]))
    row["hand_wy_direct_deg"] = _safe_float(math.degrees(hand_w_direct[1]))
    row["hand_wz_direct_deg"] = _safe_float(math.degrees(hand_w_direct[2]))
    row.update(_vec3_fields("ref", ref_pos))
    row.update(_vec3_fields("ref_v", ref_vel))
    row.update(_rad_and_deg_fields("ref_yaw", ref_yaw))
    row.update(_rad_and_deg_fields("ref_wz", ref_wz))
    row.update(_vec3_fields("rel", rel_pos))
    row.update(_vec3_fields("rel_v", rel_vel))
    row.update(_vec3_fields("pos_err", pos_err_vec))
    row.update(_vec3_fields("vel_err", vel_err_vec))
    row.update(_rad_and_deg_fields("yaw_err", yaw_err))
    row.update(_rad_and_deg_fields("w_err", w_err))
    return row


@dataclass
class ControlResult:
    action: np.ndarray
    pos_err: np.ndarray
    vel_err: np.ndarray
    yaw_err: float
    w_err: float
    vy_int: float
    vy_only_active: bool
    vy_err: float
    phase: str
    post_start_step: Optional[int]
    grasp_latched: bool


def compute_control_command(
    phase,
    ref_pos,
    ref_vel,
    grasp_center,
    ee_v,
    ref_yaw,
    ref_wz,
    front_yaw_unwrapped,
    front_wz,
    close_start_step,
    post_start_step,
    grasp_latched,
    close_schedule,
    vy_int,
    t,
    dt,
    args,
):
    """Convert the current phase references into one 7-DoF robot action."""
    action = np.zeros(7, dtype=np.float32)
    action[-1] = 1.0
    zeros = np.zeros(3, dtype=np.float32)
    result = ControlResult(
        action=action,
        pos_err=zeros.copy(),
        vel_err=zeros.copy(),
        yaw_err=0.0,
        w_err=0.0,
        vy_int=float(vy_int),
        vy_only_active=False,
        vy_err=0.0,
        phase=phase,
        post_start_step=post_start_step,
        grasp_latched=grasp_latched,
    )
    if phase == "done":
        result.action[-1] = float(args.post_hold_grip)
        return result

    kp_xyz, kv_xyz, ff_xyz = get_phase_gains(phase, args)
    u_xyz, result.pos_err, result.vel_err = control_xyz(
        ref_pos, ref_vel, grasp_center, ee_v, kp_xyz, kv_xyz, ff_xyz, max_cmd=1.0
    )
    result.action[:3] = u_xyz * get_axis_scale(phase, args)

    # Enforce only the phase-specific minimum commands required by the tuned expert.
    if phase == "descend" and float(result.pos_err[2]) < -float(args.descend_z_far_eps):
        if abs(float(result.action[2])) < float(args.descend_z_min_action):
            result.action[2] = -float(args.descend_z_min_action)

    if phase == "go_wait" and args.fast_go_wait:
        result.action[:3] = np.clip(
            result.action[:3], -float(args.go_wait_action_clip), float(args.go_wait_action_clip)
        )
        for axis in (0, 1):
            if abs(float(result.pos_err[axis])) > float(args.go_wait_far_xy):
                minimum = float(args.go_wait_min_xy_action)
                if abs(float(result.action[axis])) < minimum:
                    result.action[axis] = math.copysign(minimum, float(result.pos_err[axis]))
        if abs(float(result.pos_err[2])) > float(args.go_wait_far_z):
            minimum = float(args.go_wait_min_z_action)
            if abs(float(result.action[2])) < minimum:
                result.action[2] = math.copysign(minimum, float(result.pos_err[2]))

    if args.vy_vel_only and phase in ("launch", "descend", "close"):
        (
            result.action[1],
            result.vy_int,
            result.vy_err,
            _,
            _,
            _,
        ) = control_y_velocity_only(
            target_vy=float(ref_vel[1]),
            cur_vy=float(ee_v[1]),
            dt=float(dt),
            vy_int=float(result.vy_int),
            ee_v_per_action_y=float(args.ee_v_per_action_y),
            kv_action=float(args.vy_kv_action),
            ki_action=float(args.vy_ki_action),
            int_clip=float(args.vy_int_clip),
            action_clip=float(args.vy_action_clip),
        )
        result.vy_only_active = True

    if args.enable_yaw_sync and phase in ("go_wait", "wait_hold", "launch", "descend", "close"):
        result.yaw_err = axis_yaw_err(float(ref_yaw), float(front_yaw_unwrapped))
        result.w_err = float(ref_wz - front_wz)
        yaw_feedback = float(args.kp_yaw_sync) * result.yaw_err + float(args.kp_w_sync) * result.w_err
        yaw_feedforward = (
            float(args.yaw_ff_gain)
            * float(args.a5_per_front_wz_deg)
            * math.degrees(float(ref_wz))
        )
        result.action[5] = float(
            np.clip(yaw_feedforward - yaw_feedback, -float(args.yaw_clip), float(args.yaw_clip))
        )

    if phase == "close":
        if close_start_step is None:
            raise RuntimeError("close phase entered without close_start_step")
        schedule_index = min(
            len(close_schedule) - 1,
            (t - close_start_step) // args.close_ramp_step,
        )
        result.action[-1] = float(close_schedule[schedule_index])
        if schedule_index >= len(close_schedule) - 2:
            result.phase = "post_grasp"
            result.post_start_step = t
            result.grasp_latched = True
    elif phase == "post_grasp":
        result.action[-1] = float(args.post_hold_grip)

    if result.grasp_latched:
        result.action[-1] = float(args.post_hold_grip)
    return result


def finalize_experiment(
    args,
    log_rows,
    phase,
    grasp_latched,
    initial_cube_p,
    initial_cube_yaw,
    final_cube_p,
    final_cube_yaw,
):
    """Write detailed rows and the one-line experiment summary after control ends."""
    if args.log_csv and log_rows:
        _write_rows_csv(args.log_csv, log_rows)
        print(f"[log] wrote {len(log_rows)} rows to {args.log_csv}")

    if not args.summary_csv:
        return

    cube_disp = final_cube_p - initial_cube_p
    first_close = _first_row(
        log_rows,
        lambda row: str(row.get("phase")) in ("close", "post_grasp", "done")
        or float(row.get("grip_cmd", 1.0)) < 0.0,
    )
    first_grip = _first_row(log_rows, lambda row: float(row.get("grip_cmd", 1.0)) < 0.0)
    final_row = log_rows[-1] if log_rows else {}

    def field(row, key):
        return _safe_float(row[key]) if row is not None else float("nan")

    summary = {
        "method": args.method_name,
        "variant": args.variant,
        "yaw_state_source": args.yaw_state_source,
        "kp_w_sync": _safe_float(args.kp_w_sync),
        "wz_filter_alpha": _safe_float(args.wz_filter_alpha),
        "cube_body_to_face_deg": _safe_float(args.cube_body_to_face_deg),
        "seed": int(args.seed),
        "drift_speed": _safe_float(args.drift_speed),
        "spin_speed": _safe_float(args.spin_speed),
        "y_offset": _safe_float(args.y_offset),
        "cube_z": _safe_float(args.cube_z),
        "steps": int(args.steps),
        "final_phase": phase,
        "success": int(bool(grasp_latched or phase in ("post_grasp", "done"))),
        "entered_close": int(first_close is not None),
        "first_close_t": field(first_close, "time"),
        "first_grip_t": field(first_grip, "time"),
        "first_close_rel_speed": field(first_close, "rel_speed"),
        "first_grip_rel_speed": field(first_grip, "rel_speed"),
        "first_close_yaw_err_deg": field(first_close, "yaw_err_deg"),
        "first_close_werr_deg": field(first_close, "w_err_deg"),
        "max_abs_yaw_err_deg": _max_abs(log_rows, "yaw_err_deg"),
        "max_abs_werr_deg": _max_abs(log_rows, "w_err_deg"),
        "max_pos_err_norm": max((_safe_float(row["pos_err_norm"]) for row in log_rows), default=float("nan")),
        "max_rel_speed": max((_safe_float(row["rel_speed"]) for row in log_rows), default=float("nan")),
        "final_cube_disp_norm": _safe_float(np.linalg.norm(cube_disp)),
        "final_cube_dx": _safe_float(cube_disp[0]),
        "final_cube_dy": _safe_float(cube_disp[1]),
        "final_cube_dz": _safe_float(cube_disp[2]),
        "final_cube_yaw_change_deg": _safe_float(math.degrees(wrap_to_pi(final_cube_yaw - initial_cube_yaw))),
        "final_rel_x": _safe_float(final_row.get("rel_x", float("nan"))),
        "final_rel_y": _safe_float(final_row.get("rel_y", float("nan"))),
        "final_rel_z": _safe_float(final_row.get("rel_z", float("nan"))),
    }
    _append_summary_csv(args.summary_csv, summary)
    print(f"[summary_csv] appended to {args.summary_csv}")


def setup_environment(args):
    """Create the env, settle the scene for a few no-op steps, and tune cube physics."""
    env = create_env_and_reset(args)
    uw = env.unwrapped
    dt = uw.control_timestep
    tune_cube_physics(uw.cube, zero_damping=args.zero_damping, zero_sleep=args.zero_sleep)
    print(f"[physics] zero_damping={args.zero_damping} zero_sleep={args.zero_sleep}")

    zero = np.zeros(7, dtype=np.float32)
    zero[-1] = +1.0
    for _ in range(15):
        env.step(zero)
        if args.render:
            env.render()
    return env, uw, dt


@dataclass
class Markers:
    cube: Any = None
    grasp: Any = None
    hit: Any = None
    pre: Any = None
    ref: Any = None


def create_markers(args, scene) -> Markers:
    """Build the debug spheres, or an all-None ``Markers`` if markers are disabled."""
    if not args.show_markers:
        return Markers()
    r = args.marker_radius
    return Markers(
        cube=create_visual_marker(scene, "cube_center_marker", r, [1.0, 1.0, 0.0, 1.0]),
        grasp=create_visual_marker(scene, "grasp_center_marker", r * 0.90, [0.1, 0.9, 1.0, 1.0]),
        hit=create_visual_marker(scene, "hit_marker", r * 0.95, [1.0, 0.2, 1.0, 1.0]),
        pre=create_visual_marker(scene, "pre_marker", r * 0.85, [0.2, 1.0, 0.2, 1.0]),
        ref=create_visual_marker(scene, "ref_marker", r * 0.78, [1.0, 0.5, 0.0, 1.0]),
    )


def update_markers(markers: Markers, cube_p, grasp_center, plan: Optional[InterceptPlan], ref_pos, marker_z_offset: float) -> None:
    proj_z = float(cube_p[2] + marker_z_offset)
    set_marker_pose(markers.cube, [cube_p[0], cube_p[1], proj_z])
    set_marker_pose(markers.grasp, [grasp_center[0], grasp_center[1], proj_z])
    if plan is not None:
        set_marker_pose(markers.hit, [plan.hit_pos[0], plan.hit_pos[1], proj_z])
        set_marker_pose(markers.pre, [plan.wait_pos[0], plan.wait_pos[1], proj_z])
    set_marker_pose(markers.ref, [ref_pos[0], ref_pos[1], proj_z])


@dataclass
class LoopState:
    """Everything the control loop carries from one step to the next."""

    phase: str = "observe"
    observe_count: int = 0
    plan: Optional[InterceptPlan] = None
    close_start_step: Optional[int] = None
    post_start_step: Optional[int] = None
    prev_grasp_center: Optional[np.ndarray] = None
    cube_yaw_tracker: AngleTracker = field(default_factory=AngleTracker)
    finger_yaw_tracker: AngleTracker = field(default_factory=AngleTracker)
    hand_yaw_tracker: AngleTracker = field(default_factory=AngleTracker)
    hand_front_yaw_tracker: AngleTracker = field(default_factory=AngleTracker)
    control_wz_filtered: Optional[float] = None
    post_hold_pos: Optional[np.ndarray] = None
    post_hold_yaw: Optional[float] = None
    grasp_latched: bool = False
    vy_int: float = 0.0  # reset when entering/leaving dynamic phases
    last_phase: Optional[str] = None
    printed_plan_step: Optional[int] = None
    log_rows: list = field(default_factory=list)
    initial_cube_p: Optional[np.ndarray] = None
    initial_cube_yaw: Optional[float] = None


@dataclass
class GripperState:
    """One step's worth of gripper geometry, yaw estimate, and derived control point."""

    axis_point: np.ndarray
    axis_source_name: str
    finger_mid: np.ndarray
    left_finger_p: np.ndarray
    right_finger_p: np.ndarray
    grasp_center: np.ndarray
    ee_v: np.ndarray
    finger_axis_yaw: float
    finger_axis_wz: float
    hand_roll: float
    hand_pitch: float
    hand_yaw: float
    hand_yaw_unwrapped: float
    hand_yaw_wz_fd: float
    hand_w_direct: np.ndarray
    hand_w_source: str
    front_yaw: float
    front_yaw_unwrapped: float
    front_wz: float
    control_yaw_source: str
    use_hand_state: bool


def read_cube_state(uw, state: LoopState, dt: float):
    """Return (cube_p, cube_v, cube_yaw, cube_wz) and update the yaw tracker in state."""
    cube_p = np_pose(uw.cube.pose.p)
    cube_q = np_pose(uw.cube.pose.q)
    cube_yaw = float(quat_to_euler_xyz_wxyz(cube_q)[2])
    cube_v = actor_linear_velocity(uw.cube)
    _, cube_wz = state.cube_yaw_tracker.update(cube_yaw, dt)
    return cube_p, cube_v, cube_yaw, cube_wz


def read_gripper_state(uw, args, state: LoopState, dt: float) -> GripperState:
    """Read gripper geometry/yaw for one step and advance the trackers in state.

    Also derives the hybrid grasp control point (hand-axis XY, finger-mid Z for
    clearance) and its finite-difference velocity, since both are downstream of
    the same geometry read.
    """
    (
        axis_point,
        axis_source_name,
        finger_mid,
        axis_open,
        left_finger_p,
        right_finger_p,
        hand_link,
    ) = read_gripper_geometry(uw, axis_z_offset=args.axis_z_offset)
    finger_axis_yaw = wrap_to_pi(
        yaw_from_front_normal(axis_open) + math.radians(args.gripper_front_bias_deg)
    )
    finger_yaw_unwrapped, finger_axis_wz = state.finger_yaw_tracker.update(finger_axis_yaw, dt)

    hand_q = np_pose(hand_link.pose.q)
    hand_rpy = quat_to_euler_xyz_wxyz(hand_q)
    hand_roll, hand_pitch, hand_yaw = float(hand_rpy[0]), float(hand_rpy[1]), float(hand_rpy[2])
    hand_yaw_unwrapped, hand_yaw_wz_fd = state.hand_yaw_tracker.update(hand_yaw, dt)
    hand_w_direct = actor_angular_velocity(hand_link)
    hand_w_source = "link.angular_velocity"

    # The rigid hand local axis avoids the finger-origin projection's yaw bias.
    use_hand_state = args.yaw_state_source == "hand"
    if use_hand_state:
        hand_front_yaw = projected_local_axis_yaw_wxyz(
            hand_q, axis_index=0, bias_deg=float(args.gripper_front_bias_deg)
        )
        hand_front_yaw_unwrapped, _ = state.hand_front_yaw_tracker.update(hand_front_yaw, dt)
        state.control_wz_filtered = low_pass(
            state.control_wz_filtered, float(hand_w_direct[2]), float(args.wz_filter_alpha)
        )
        front_yaw = float(hand_front_yaw)
        front_yaw_unwrapped = float(hand_front_yaw_unwrapped)
        front_wz = float(state.control_wz_filtered)
        control_yaw_source = "panda_hand"
    else:
        front_yaw = float(finger_axis_yaw)
        front_yaw_unwrapped = float(finger_yaw_unwrapped)
        front_wz = float(finger_axis_wz)
        control_yaw_source = "finger_axis"

    # Hybrid point: hand-axis XY for alignment, physical finger-mid Z for clearance.
    grasp_center = np.array([axis_point[0], axis_point[1], finger_mid[2]], dtype=np.float32)
    if state.prev_grasp_center is None:
        ee_v = np.zeros(3, dtype=np.float32)
    else:
        ee_v = ((grasp_center - state.prev_grasp_center) / dt).astype(np.float32)
    state.prev_grasp_center = grasp_center.copy()

    return GripperState(
        axis_point=axis_point,
        axis_source_name=axis_source_name,
        finger_mid=finger_mid,
        left_finger_p=left_finger_p,
        right_finger_p=right_finger_p,
        grasp_center=grasp_center,
        ee_v=ee_v,
        finger_axis_yaw=finger_axis_yaw,
        finger_axis_wz=finger_axis_wz,
        hand_roll=hand_roll,
        hand_pitch=hand_pitch,
        hand_yaw=hand_yaw,
        hand_yaw_unwrapped=hand_yaw_unwrapped,
        hand_yaw_wz_fd=hand_yaw_wz_fd,
        hand_w_direct=hand_w_direct,
        hand_w_source=hand_w_source,
        front_yaw=front_yaw,
        front_yaw_unwrapped=front_yaw_unwrapped,
        front_wz=front_wz,
        control_yaw_source=control_yaw_source,
        use_hand_state=use_hand_state,
    )


def decide_phase(phase: str, plan: InterceptPlan, now_t: float, grasp_center: np.ndarray, args) -> str:
    """Pick the next phase from the plan's timeline; a no-op once post_grasp/done is reached."""
    if phase in ("post_grasp", "done"):
        return phase
    if now_t >= plan.close_start_time:
        return "close"
    if now_t >= plan.descend_start_time:
        return "descend"
    if now_t >= plan.lin_start_time:
        return "launch"
    wait_err_xy = float(np.linalg.norm((plan.wait_pos[:2] - grasp_center[:2]).astype(np.float64)))
    wait_err_z = abs(float(plan.wait_pos[2] - grasp_center[2]))
    at_wait = wait_err_xy <= args.go_wait_xy_eps and wait_err_z <= args.go_wait_z_eps
    return "wait_hold" if at_wait else "go_wait"


def format_debug_line(
    *,
    t: int,
    phase: str,
    cube_p,
    cube_v,
    gs: GripperState,
    ref_pos,
    ref_vel,
    ref_yaw: float,
    ref_wz: float,
    yaw_err: float,
    cube_wz: float,
    w_err: float,
    pos_err_vec,
    vel_err_vec,
    plan_tau: float,
    plan_hit: float,
    time_to_hit,
    launch_s: float,
    action,
    vy_only_active: bool,
    vy_err_only: float,
    vy_int: float,
) -> str:
    grasp_center, axis_point, finger_mid = gs.grasp_center, gs.axis_point, gs.finger_mid
    left_finger_p, right_finger_p, ee_v = gs.left_finger_p, gs.right_finger_p, gs.ee_v
    return (
        f"[t={t:03d}] phase={phase:<11s} "
        f"cube=({cube_p[0]:+.3f},{cube_p[1]:+.3f},{cube_p[2]:+.3f}) "
        f"ctrl_pt=({grasp_center[0]:+.3f},{grasp_center[1]:+.3f},{grasp_center[2]:+.3f}) "
        f"axis_pt=({axis_point[0]:+.3f},{axis_point[1]:+.3f},{axis_point[2]:+.3f}) "
        f"finger_mid=({finger_mid[0]:+.3f},{finger_mid[1]:+.3f},{finger_mid[2]:+.3f}) "
        f"Lrel=({left_finger_p[0]-cube_p[0]:+.3f},{left_finger_p[1]-cube_p[1]:+.3f},{left_finger_p[2]-cube_p[2]:+.3f}) "
        f"Rrel=({right_finger_p[0]-cube_p[0]:+.3f},{right_finger_p[1]-cube_p[1]:+.3f},{right_finger_p[2]-cube_p[2]:+.3f}) "
        f"ctrl_rel=({grasp_center[0]-cube_p[0]:+.3f},{grasp_center[1]-cube_p[1]:+.3f},{grasp_center[2]-cube_p[2]:+.3f}) "
        f"axis_rel=({axis_point[0]-cube_p[0]:+.3f},{axis_point[1]-cube_p[1]:+.3f},{axis_point[2]-cube_p[2]:+.3f}) "
        f"mid_rel=({finger_mid[0]-cube_p[0]:+.3f},{finger_mid[1]-cube_p[1]:+.3f},{finger_mid[2]-cube_p[2]:+.3f}) "
        f"ref=({ref_pos[0]:+.3f},{ref_pos[1]:+.3f},{ref_pos[2]:+.3f}) "
        f"cube_v=({cube_v[0]:+.3f},{cube_v[1]:+.3f},{cube_v[2]:+.3f}) ee_v=({ee_v[0]:+.3f},{ee_v[1]:+.3f},{ee_v[2]:+.3f}) "
        f"ref_v=({ref_vel[0]:+.3f},{ref_vel[1]:+.3f},{ref_vel[2]:+.3f}) "
        f"front_yaw={math.degrees(gs.front_yaw):+.2f} ref_yaw={math.degrees(ref_yaw):+.2f} yaw_err={math.degrees(yaw_err):+.2f} "
        f"cube_wz={math.degrees(cube_wz):+.2f} ref_wz={math.degrees(ref_wz):+.2f} werr={math.degrees(w_err):+.2f} "
        f"pos_err=({pos_err_vec[0]:+.3f},{pos_err_vec[1]:+.3f},{pos_err_vec[2]:+.3f}) vel_err=({vel_err_vec[0]:+.3f},{vel_err_vec[1]:+.3f},{vel_err_vec[2]:+.3f}) "
        f"plan_tau={plan_tau:+.3f} hit_t={plan_hit:+.3f} t_hit={time_to_hit if time_to_hit is not None else float('nan'):+.3f} launch_s={launch_s:+.2f} "
        f"a_xyz=({action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f}) a5={action[5]:+.3f} grip={action[-1]:+.2f} "
        f"vy_only={int(vy_only_active)} vy_err={vy_err_only:+.3f} vy_int={vy_int:+.3f} "
        f"axis_src={gs.axis_source_name}"
    )


@dataclass
class EpisodeResult:
    phase: str
    grasp_latched: bool
    log_rows: list
    initial_cube_p: np.ndarray
    initial_cube_yaw: float
    final_cube_p: np.ndarray
    final_cube_yaw: float
    axis_source_name: str


def run_episode(env, uw, dt: float, args, markers: Markers) -> EpisodeResult:
    """Run the observe -> plan -> go_wait -> ... -> done control loop for one episode."""
    state = LoopState()
    close_schedule = build_close_schedule()

    print("\n=== Moving and rotating target grasping expert ===")
    print(
        f"[init] drift_speed={args.drift_speed:.3f} y_offset={args.y_offset:.3f} spin_speed={args.spin_speed:.3f} cube_z={args.cube_z:.3f} "
        f"observe_steps={args.observe_steps} hover_clear={args.observe_hover_clearance:.3f} closest_tau=[{args.closest_tau_min:.2f},{args.closest_tau_max:.2f}] launch_time={args.launch_time:.3f}"
    )
    print(
        f"[experiment] method={args.method_name} variant={args.variant} seed={args.seed} "
        f"y_offset={args.y_offset:+.3f} log_csv={args.log_csv or 'None'} "
        f"summary_csv={args.summary_csv or 'None'}"
    )

    for t in range(args.steps):
        now_t = t * dt
        cube_p, cube_v, cube_yaw, cube_wz = read_cube_state(uw, state, dt)
        gs = read_gripper_state(uw, args, state, dt)

        if state.phase == "observe":
            state.observe_count += 1
            if state.observe_count >= args.observe_steps:
                state.phase = "plan"

        if state.phase == "plan":
            state.plan = choose_intercept_plan(
                step_idx=t,
                now_t=now_t,
                cube_p=cube_p,
                cube_v=cube_v,
                cube_yaw=cube_yaw,
                cube_wz=cube_wz,
                grasp_center=gs.grasp_center,
                front_yaw_unwrapped=gs.front_yaw_unwrapped,
                args=args,
            )
            state.close_start_step = None
            state.post_start_step = None
            state.post_hold_pos = None
            state.post_hold_yaw = None
            state.grasp_latched = False
            state.vy_int = 0.0
            state.phase = "go_wait"
            if state.printed_plan_step != state.plan.plan_step:
                print_plan_debug(state.plan, grasp_center=gs.grasp_center, cube_p=cube_p, cube_v=cube_v)
                state.printed_plan_step = state.plan.plan_step

        plan = state.plan
        time_to_hit = None if plan is None else float(plan.hit_time - now_t)
        ref_pos = gs.grasp_center.copy()
        ref_vel = np.zeros(3, dtype=np.float32)
        ref_yaw = gs.front_yaw
        ref_wz = 0.0
        launch_s = 0.0

        if state.phase == "observe":
            ref_pos = gs.grasp_center.copy()
            ref_pos[2] = max(float(gs.grasp_center[2]), float(cube_p[2] + args.observe_hover_clearance))
            ref_vel = np.zeros(3, dtype=np.float32)

        if plan is not None:
            state.phase = decide_phase(state.phase, plan, now_t, gs.grasp_center, args)
            if state.phase == "close" and state.close_start_step is None:
                state.close_start_step = t

            if state.phase != state.last_phase:
                if state.phase in ("go_wait", "wait_hold", "launch", "descend", "close", "post_grasp"):
                    print_phase_debug(
                        tag=state.phase,
                        now_t=now_t,
                        plan=plan,
                        grasp_center=gs.grasp_center,
                        ee_v=gs.ee_v,
                        front_yaw=gs.front_yaw,
                        front_wz=gs.front_wz,
                    )
                    if state.phase in ("launch", "go_wait", "wait_hold", "post_grasp"):
                        state.vy_int = 0.0
                state.last_phase = state.phase

            if state.phase in ("go_wait", "wait_hold"):
                ref_pos = plan.wait_pos.copy()
                ref_vel = np.zeros(3, dtype=np.float32)
            elif state.phase in ("launch", "descend"):
                ref_pos, ref_vel, launch_s = compute_launch_descend_reference(plan, now_t, args)

        if state.phase == "close" and plan is not None:
            ref_pos = plan.close_pos.copy()
            if args.variant == "predictive_intercept":
                # Stop at the predicted interception point: no terminal velocity sync.
                ref_vel = np.zeros(3, dtype=np.float32)
            else:
                ref_vel = plan.target_vel.copy()
                ref_vel[2] = 0.0

        if state.phase == "post_grasp" and plan is not None:
            if state.post_hold_pos is None:
                state.post_hold_pos = gs.grasp_center.copy()
                state.post_hold_yaw = float(gs.front_yaw)
            ref_pos = state.post_hold_pos.copy()
            ref_vel = np.zeros(3, dtype=np.float32)
            if state.post_start_step is not None and (t - state.post_start_step) >= args.post_steps:
                state.phase = "done"

        ref_yaw, ref_wz = compute_yaw_reference(state.phase, plan, now_t, gs.front_yaw, state.post_hold_yaw, args)
        update_markers(markers, cube_p, gs.grasp_center, plan, ref_pos, args.marker_z_offset)

        control = compute_control_command(
            state.phase,
            ref_pos,
            ref_vel,
            gs.grasp_center,
            gs.ee_v,
            ref_yaw,
            ref_wz,
            gs.front_yaw_unwrapped,
            gs.front_wz,
            state.close_start_step,
            state.post_start_step,
            state.grasp_latched,
            close_schedule,
            state.vy_int,
            t,
            dt,
            args,
        )
        action = control.action
        pos_err_vec, vel_err_vec = control.pos_err, control.vel_err
        yaw_err, w_err = control.yaw_err, control.w_err
        state.vy_int = control.vy_int
        vy_only_active, vy_err_only = control.vy_only_active, control.vy_err
        state.phase = control.phase
        state.post_start_step = control.post_start_step
        state.grasp_latched = control.grasp_latched

        plan_tau = float(plan.tau) if plan is not None else float("nan")
        plan_hit = float(plan.hit_time) if plan is not None else float("nan")

        if state.initial_cube_p is None:
            state.initial_cube_p = cube_p.copy()
            state.initial_cube_yaw = float(cube_yaw)

        if args.log_csv and (t % args.log_every == 0 or t == args.steps - 1):
            state.log_rows.append(build_step_log_row(
                args,
                t=t,
                now_t=now_t,
                phase=state.phase,
                cube_p=cube_p,
                cube_v=cube_v,
                cube_yaw=cube_yaw,
                cube_wz=cube_wz,
                grasp_center=gs.grasp_center,
                ee_v=gs.ee_v,
                front_yaw=gs.front_yaw,
                front_yaw_unwrapped=gs.front_yaw_unwrapped,
                front_wz=gs.front_wz,
                control_yaw_source=gs.control_yaw_source,
                use_hand_state=gs.use_hand_state,
                finger_axis_yaw=gs.finger_axis_yaw,
                finger_axis_wz=gs.finger_axis_wz,
                hand_roll=gs.hand_roll,
                hand_pitch=gs.hand_pitch,
                hand_yaw=gs.hand_yaw,
                hand_yaw_unwrapped=gs.hand_yaw_unwrapped,
                hand_yaw_wz_fd=gs.hand_yaw_wz_fd,
                hand_w_direct=gs.hand_w_direct,
                hand_w_source=gs.hand_w_source,
                hand_link_name_diag=gs.axis_source_name,
                ref_pos=ref_pos,
                ref_vel=ref_vel,
                ref_yaw=ref_yaw,
                ref_wz=ref_wz,
                pos_err_vec=pos_err_vec,
                vel_err_vec=vel_err_vec,
                yaw_err=yaw_err,
                w_err=w_err,
                action=action,
                plan_tau=plan_tau,
                plan_hit=plan_hit,
                time_to_hit=time_to_hit,
                launch_s=launch_s,
                close_start_step=state.close_start_step,
                post_start_step=state.post_start_step,
                grasp_latched=state.grasp_latched,
                vy_only_active=vy_only_active,
                vy_err_only=vy_err_only,
            ))

        if args.keep_awake_every_step:
            wake_cube(uw.cube)
        env.step(action)
        if args.keep_awake_every_step:
            wake_cube(uw.cube)
        if args.render:
            env.render()

        if t % args.debug_every == 0 or t == args.steps - 1:
            print(format_debug_line(
                t=t,
                phase=state.phase,
                cube_p=cube_p,
                cube_v=cube_v,
                gs=gs,
                ref_pos=ref_pos,
                ref_vel=ref_vel,
                ref_yaw=ref_yaw,
                ref_wz=ref_wz,
                yaw_err=yaw_err,
                cube_wz=cube_wz,
                w_err=w_err,
                pos_err_vec=pos_err_vec,
                vel_err_vec=vel_err_vec,
                plan_tau=plan_tau,
                plan_hit=plan_hit,
                time_to_hit=time_to_hit,
                launch_s=launch_s,
                action=action,
                vy_only_active=vy_only_active,
                vy_err_only=vy_err_only,
                vy_int=state.vy_int,
            ))

    return EpisodeResult(
        phase=state.phase,
        grasp_latched=state.grasp_latched,
        log_rows=state.log_rows,
        initial_cube_p=state.initial_cube_p,
        initial_cube_yaw=state.initial_cube_yaw,
        final_cube_p=cube_p.copy(),
        final_cube_yaw=float(cube_yaw),
        axis_source_name=gs.axis_source_name,
    )


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    validate_args(args)

    args.method_name = resolve_method_name(args)
    print(f"[ablation] variant={args.variant} method_name={args.method_name}")

    env, uw, dt = setup_environment(args)
    markers = create_markers(args, uw.scene)

    result = run_episode(env, uw, dt, args, markers)

    finalize_experiment(
        args,
        result.log_rows,
        result.phase,
        result.grasp_latched,
        result.initial_cube_p,
        result.initial_cube_yaw,
        result.final_cube_p,
        result.final_cube_yaw,
    )
    print(
        f"\n[summary] finished variant={args.variant} final_phase={result.phase} "
        f"grasp_latched={int(result.grasp_latched)} yaw_source={args.yaw_state_source} "
        f"axis_source={result.axis_source_name}"
    )
    env.close()


if __name__ == "__main__":
    main()

