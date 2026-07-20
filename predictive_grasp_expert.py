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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien  # noqa: F401
import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose


# ---------- generic actor helpers ----------
def iter_targets(actor: Any):
    yield ("actor", actor)
    for attr in ["_objs", "_bodies", "objs", "bodies"]:
        obj = getattr(actor, attr, None)
        if obj is None:
            continue
        try:
            items = list(obj)
        except Exception:
            continue
        for i, item in enumerate(items):
            yield (f"{attr}[{i}]", item)


def call_first_success(actor: Any, method_names: Sequence[str], *args) -> Optional[str]:
    for target_name, target in iter_targets(actor):
        for method_name in method_names:
            fn = getattr(target, method_name, None)
            if fn is None:
                continue
            try:
                fn(*args)
                return f"{target_name}.{method_name}{tuple(args)}"
            except Exception:
                continue
    return None


def try_wake(actor: Any) -> Optional[str]:
    hit = call_first_success(actor, ["wake_up", "wakeUp"])
    if hit is not None:
        return hit
    return call_first_success(actor, ["set_wake_counter", "setWakeCounter"], 1e6)


def tune_cube_physics(actor: Any, zero_damping: bool = True, zero_sleep: bool = True):
    hits = []
    if zero_damping:
        for names, args in [
            (["set_linear_damping", "setLinearDamping"], (0.0,)),
            (["set_angular_damping", "setAngularDamping"], (0.0,)),
            (["set_damping", "setDamping"], (0.0, 0.0)),
        ]:
            hit = call_first_success(actor, names, *args)
            if hit is not None:
                hits.append(hit)
    if zero_sleep:
        for names, args in [
            (["set_sleep_threshold", "setSleepThreshold", "set_sleep_thresh"], (0.0,)),
            (["set_stabilization_threshold", "setStabilizationThreshold"], (0.0,)),
            (["set_wake_counter", "setWakeCounter"], (1e6,)),
        ]:
            hit = call_first_success(actor, names, *args)
            if hit is not None:
                hits.append(hit)
        wake_hit = try_wake(actor)
        if wake_hit is not None:
            hits.append(wake_hit)
    return hits


# ---------- marker helpers ----------


    def maybe_yield(obj):
        if obj is None:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        yield obj

    for name in direct_names:
        obj = getattr(uw, name, None)
        for x in maybe_yield(obj):
            yield x
        if obj is None:
            continue
        for nested in nested_names:
            items = getattr(obj, nested, None)
            if items is None:
                continue
            try:
                for it in list(items):
                    for x in maybe_yield(it):
                        yield x
            except Exception:
                continue

    for nested in nested_names:
        items = getattr(uw, nested, None)
        if items is None:
            continue
        try:
            for it in list(items):
                for x in maybe_yield(it):
                    yield x
        except Exception:
            continue


def _make_render_material(color_rgba):
    mat = None
    try:
        mat = sapien.render.RenderMaterial()
        rgba = np.array(color_rgba, dtype=np.float32)
        if hasattr(mat, "base_color"):
            mat.base_color = rgba
        elif hasattr(mat, "set_base_color"):
            mat.set_base_color(rgba)
    except Exception:
        mat = None
    return mat


def _build_marker_actor(builder, name: str):
    for method_name in ["build_kinematic", "build_static", "build"]:
        fn = getattr(builder, method_name, None)
        if fn is None:
            continue
        try:
            return fn(name=name)
        except TypeError:
            try:
                return fn()
            except Exception:
                continue
        except Exception:
            continue
    return None


def create_visual_marker(uw, name: str, radius: float, color_rgba):
    mat = _make_render_material(color_rgba)
    errs = []
    for sc in _iter_scene_candidates(uw):
        builder_fn = getattr(sc, "create_actor_builder", None)
        if builder_fn is None:
            continue
        try:
            builder = builder_fn()
        except Exception as e:
            errs.append(f"create_actor_builder failed: {e}")
            continue

        added = False
        for visual_name in ["add_sphere_visual", "add_box_visual"]:
            vf = getattr(builder, visual_name, None)
            if vf is None:
                continue
            try:
                if visual_name == "add_sphere_visual":
                    if mat is not None:
                        vf(radius=radius, material=mat)
                    else:
                        vf(radius=radius)
                else:
                    half = [radius, radius, radius]
                    if mat is not None:
                        vf(half_size=half, material=mat)
                    else:
                        vf(half_size=half)
                added = True
                break
            except TypeError:
                try:
                    if visual_name == "add_sphere_visual":
                        vf(radius)
                    else:
                        vf([radius, radius, radius])
                    added = True
                    break
                except Exception as e:
                    errs.append(f"{visual_name} fallback failed: {e}")
            except Exception as e:
                errs.append(f"{visual_name} failed: {e}")

        if not added:
            continue

        actor = _build_marker_actor(builder, name)
        if actor is not None:
            return actor, "ok"

    return None, "; ".join(errs[:3]) if errs else "no scene/create_actor_builder available"


def set_marker_pose(marker: Any, p_xyz):
    if marker is None:
        return False
    p_xyz = np.array(p_xyz, dtype=np.float32)
    for pose_obj in [
        lambda: sapien.Pose(p=p_xyz),
        lambda: sapien.Pose(p_xyz),
        lambda: Pose.create_from_pq(p=p_xyz, q=[1, 0, 0, 0]),
    ]:
        try:
            pose = pose_obj()
        except Exception:
            continue
        try:
            marker.set_pose(pose)
            return True
        except Exception:
            continue
    return False


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



def np_pose(x):
    try:
        return x[0].cpu().numpy()
    except Exception:
        return np.array(x, dtype=np.float32)


def np_vec3(x):
    arr = np.array(x, dtype=np.float32).reshape(-1)
    if arr.size >= 3:
        return arr[:3].astype(np.float32)
    out = np.zeros(3, dtype=np.float32)
    out[:arr.size] = arr.astype(np.float32)
    return out


def try_get_actor_vec3(actor: Any, attr_names: Sequence[str], method_names: Sequence[str]):
    for target_name, target in iter_targets(actor):
        for attr_name in attr_names:
            try:
                val = getattr(target, attr_name)
            except Exception:
                continue
            if callable(val):
                continue
            try:
                return np_vec3(np_pose(val)), f"{target_name}.{attr_name}"
            except Exception:
                continue
        for method_name in method_names:
            fn = getattr(target, method_name, None)
            if fn is None:
                continue
            try:
                return np_vec3(np_pose(fn())), f"{target_name}.{method_name}()"
            except Exception:
                continue
    return None, None


def get_actor_linear_velocity(actor: Any, prev_p=None, dt: Optional[float] = None):
    v, src = try_get_actor_vec3(
        actor,
        attr_names=["linear_velocity", "linvel", "velocity", "v"],
        method_names=["get_linear_velocity", "getLinearVelocity"],
    )
    if v is not None:
        return v.astype(np.float32), src
    if prev_p is not None and dt is not None and dt > 1e-8:
        return ((np_pose(actor.pose.p) - prev_p) / dt).astype(np.float32), "finite_diff"
    return np.zeros(3, dtype=np.float32), "zero"


def get_actor_angular_velocity(actor: Any):
    """Read a link/actor world-frame angular velocity without finite differencing.

    ManiSkill/SAPIEN wrappers expose this under slightly different attribute or
    method names across versions, so reuse the generic actor traversal helper.
    Returns (omega_xyz, source).  NaNs mean no direct source was available.
    """
    w, src = try_get_actor_vec3(
        actor,
        attr_names=["angular_velocity", "angvel", "omega", "w"],
        method_names=["get_angular_velocity", "getAngularVelocity"],
    )
    if w is not None:
        return w.astype(np.float32), src
    return np.full(3, np.nan, dtype=np.float32), "unavailable"


def wrap_to_pi(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


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
    return None, "raw_mid_fallback", "raw_mid"


def make_planar_axes(axis_open):
    axis_lat = np.array([axis_open[0], axis_open[1]], dtype=np.float32)
    n = np.linalg.norm(axis_lat)
    if n < 1e-8:
        axis_lat = np.array([0.0, 1.0], dtype=np.float32)
    else:
        axis_lat = axis_lat / n
    axis_fwd = np.array([-axis_lat[1], axis_lat[0]], dtype=np.float32)
    return axis_lat, axis_fwd




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






def get_face_center_raw_mid_and_open_axis(uw, face_forward_offset=0.018, face_down_offset=-0.022):
    lf, rf = find_finger_links(uw)
    lp = np_pose(lf.pose.p)
    rp = np_pose(rf.pose.p)
    raw_mid = 0.5 * (lp + rp)

    axis_open = rp - lp
    n = np.linalg.norm(axis_open)
    if n < 1e-8:
        axis_open = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        axis_open = axis_open / n

    axis_lat, axis_fwd = make_planar_axes(axis_open)
    face_center = raw_mid.copy()
    face_center[:2] += float(face_forward_offset) * axis_fwd
    face_center[2] += float(face_down_offset)
    finger_min_z = float(min(lp[2], rp[2]))
    return face_center.astype(np.float32), raw_mid.astype(np.float32), axis_open.astype(np.float32), finger_min_z


def get_yaw_axis_point(uw, axis_z_offset=0.0, face_forward_offset=0.018, face_down_offset=-0.022):
    face_center, raw_mid, axis_open, finger_min_z = get_face_center_raw_mid_and_open_axis(
        uw, face_forward_offset=face_forward_offset, face_down_offset=face_down_offset
    )
    hand_link, source_name, source_kind = find_hand_like_link(uw)
    if hand_link is None:
        axis_point = raw_mid.copy()
    else:
        axis_point = np_pose(hand_link.pose.p).copy()
    axis_point[2] += float(axis_z_offset)
    return axis_point.astype(np.float32), source_name, source_kind, face_center, raw_mid, axis_open, finger_min_z
















@dataclass
class InterceptPlan:
    plan_step: int
    plan_time: float
    wait_time: float
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
    score: float
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
    locked_face_offset: Optional[float] = None,
):
    """Predictive translation plan plus a z-yaw/wz reference.

    Predict a planar intercept and a z-yaw/angular-rate reference. For rotation, choose a cube face
    axis at hit time, then backsolve a wait yaw so that launch/descend/close can
    spin with cube_wz and arrive near hit_yaw at hit_time.
    """
    del locked_face_offset

    v_xy = cube_v[:2].astype(np.float64)
    speed = float(np.linalg.norm(v_xy))
    if speed < 1e-8:
        dir_xy = np.array([0.0, 1.0], dtype=np.float64)
    else:
        dir_xy = v_xy / speed

    tau_grid = np.arange(args.closest_tau_min, args.closest_tau_max + 0.5 * args.plan_t_step, args.plan_t_step)
    if tau_grid.size == 0:
        tau_grid = np.array([args.closest_tau_min], dtype=np.float64)

    lin_total_time = max(
        float(args.launch_time),
        speed / max(1e-6, float(args.xy_accel_budget)) + float(args.lin_time_margin),
    )
    launch_backtrack_s, lin_accel_time = linear_backtrack_distance(
        speed=speed,
        accel_abs=float(args.xy_accel_budget),
        total_time=lin_total_time,
    )
    # IMPORTANT: in the translation-only schedule, the launch profile must exactly
    # cover the backtracked distance by t_hit.  An extra lag offset here would make
    # wait_pos farther back than the open-loop launch profile can actually travel,
    # which guarantees a systematic y miss even if execution perfectly follows time.
    # So this version intentionally does NOT add any extra xy lag compensation.

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

        # Select the intercept from planar distance; Z feasibility is handled separately.
        # Do not let the separate Z clearance objective affect which future XY intercept
        # point is selected. Z feasibility is already handled by wait_z_dist / est_wait_time.
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

    # Predict the cube face yaw at hit time. During go_wait /
    # wait_hold we rotate toward the backsolved wait_yaw; during launch /
    # descend / close the reference yaw advances at target_wz so it reaches
    # hit_yaw at hit_time.  The requested turn is softly capped to about 100 deg
    # by default to avoid choosing a long unnecessary spin.
    face_offsets = [math.radians(float(args.cube_body_to_face_deg)),
                    math.radians(float(args.cube_body_to_face_deg)) + math.pi / 2.0]
    cube_yaw_hit = float(cube_yaw + cube_wz * tau)
    _, yaw_face_offset, hit_yaw, wait_yaw, yaw_delta_wait = choose_face_target_yaw(
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
        wait_time=float(lin_start_time),
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
        score=float(score),
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
    try:
        return float(x)
    except Exception:
        return float("nan")


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
        try:
            if predicate(row):
                return row
        except Exception:
            continue
    return None


def _max_abs(rows, key):
    vals = []
    for row in rows:
        v = row.get(key, float("nan"))
        try:
            if math.isfinite(float(v)):
                vals.append(abs(float(v)))
        except Exception:
            pass
    return max(vals) if vals else float("nan")



def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="State-based expert for a translating and rotating cube.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--debug_every", type=int, default=2)

    ap.add_argument("--drift_speed", type=float, default=0.03)
    ap.add_argument("--y_offset", type=float, default=-0.60)
    ap.add_argument("--spin_speed", type=float, default=0.30)
    ap.add_argument("--cube_z", type=float, default=0.14)

    # Experiment / logging options. These do not change the control logic.
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

    ap.add_argument("--zero_damping", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--zero_sleep", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--keep_awake_every_step", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--move_scale", type=float, default=0.24)
    # Fast pre-positioning controller for go_wait. The normal PD controller is
    # accurate but slows down asymptotically near wait_pos, which leaves too little
    # time to wait/stabilize when drift_speed is high. These parameters only affect
    # go_wait; launch/descend/close keep the tuned velocity-synchronization logic.
    ap.add_argument("--fast_go_wait", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--go_wait_move_scale", type=float, default=0.42)
    ap.add_argument("--go_wait_action_clip", type=float, default=0.42)
    ap.add_argument("--go_wait_min_xy_action", type=float, default=0.055)
    ap.add_argument("--go_wait_min_z_action", type=float, default=0.025)
    ap.add_argument("--go_wait_far_xy", type=float, default=0.025)
    ap.add_argument("--go_wait_far_z", type=float, default=0.012)
    ap.add_argument("--launch_y_action_scale", type=float, default=1.25)
    ap.add_argument("--descend_y_action_scale", type=float, default=1.20)
    ap.add_argument("--descend_z_action_scale", type=float, default=2.35)
    ap.add_argument("--descend_z_min_action", type=float, default=0.055)
    ap.add_argument("--descend_z_far_eps", type=float, default=0.006)

    # During launch/descend/close, replace y position tracking with pure
    # y-velocity servo. go_wait/wait_hold still use position control so the
    # gripper can reach the precomputed wait point.
    ap.add_argument("--vy_vel_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ee_v_per_action_y", type=float, default=0.75)
    ap.add_argument("--vy_kv_action", type=float, default=1.20)
    ap.add_argument("--vy_ki_action", type=float, default=0.80)
    ap.add_argument("--vy_int_clip", type=float, default=0.05)
    ap.add_argument("--vy_action_clip", type=float, default=0.14)

    ap.add_argument("--axis_z_offset", type=float, default=0.0)
    ap.add_argument("--face_forward_offset", type=float, default=0.018)
    ap.add_argument("--face_down_offset", type=float, default=-0.022)
    ap.add_argument("--gripper_front_bias_deg", type=float, default=-2.2)
    ap.add_argument("--cube_body_to_face_deg", type=float, default=0.0)

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

    ap.add_argument("--kp_xy_pre", type=float, default=4.0)
    ap.add_argument("--kp_z_pre", type=float, default=3.0)

    ap.add_argument("--kp_xy_sync", type=float, default=3.8)
    ap.add_argument("--kp_z_sync", type=float, default=2.4)
    ap.add_argument("--kv_xy_sync", type=float, default=1.0)
    ap.add_argument("--kv_z_sync", type=float, default=0.5)
    ap.add_argument("--ff_xy_sync", type=float, default=0.45)
    ap.add_argument("--ff_z_sync", type=float, default=0.00)
    ap.add_argument("--launch_kp_x_scale", type=float, default=0.60)
    ap.add_argument("--launch_kp_y_scale", type=float, default=1.35)
    ap.add_argument("--launch_kv_x_scale", type=float, default=0.55)
    ap.add_argument("--launch_kv_y_scale", type=float, default=1.60)
    ap.add_argument("--launch_ff_x_scale", type=float, default=1.00)
    ap.add_argument("--launch_ff_y_scale", type=float, default=1.85)
    ap.add_argument("--descend_kp_x_scale", type=float, default=0.75)
    ap.add_argument("--descend_kp_y_scale", type=float, default=1.55)
    ap.add_argument("--descend_kp_z_scale", type=float, default=3.20)
    ap.add_argument("--descend_kv_x_scale", type=float, default=0.75)
    ap.add_argument("--descend_kv_y_scale", type=float, default=1.70)
    ap.add_argument("--descend_kv_z_scale", type=float, default=1.10)
    ap.add_argument("--descend_ff_x_scale", type=float, default=1.00)
    ap.add_argument("--descend_ff_y_scale", type=float, default=1.75)
    ap.add_argument("--descend_ff_z_scale", type=float, default=1.75)
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

    ap.add_argument("--close_ramp_step", type=int, default=3)
    ap.add_argument("--go_wait_xy_eps", type=float, default=0.012)
    ap.add_argument("--go_wait_z_eps", type=float, default=0.010)
    ap.add_argument("--post_hold_grip", type=float, default=-0.85)
    ap.add_argument("--post_steps", type=int, default=40)
    ap.add_argument("--post_ff_xy", type=float, default=0.45)

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
    return (
        np.array([args.kp_xy_sync, args.kp_xy_sync, args.kp_z_sync], dtype=np.float32),
        np.array([args.kv_xy_sync, args.kv_xy_sync, args.kv_z_sync], dtype=np.float32),
        np.array([args.post_ff_xy, args.post_ff_xy, 0.0], dtype=np.float32),
    )


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


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    args.method_name = resolve_method_name(args)
    print(f"[ablation] variant={args.variant} method_name={args.method_name}")

    env = create_env_and_reset(args)

    uw = env.unwrapped
    dt = getattr(uw, "control_dt", None) or getattr(uw, "control_timestep", None) or (1.0 / 20.0)
    physics_hits = tune_cube_physics(uw.cube, zero_damping=args.zero_damping, zero_sleep=args.zero_sleep)



    zero = np.zeros(7, dtype=np.float32)
    zero[-1] = +1.0
    for _ in range(15):
        env.step(zero)
        if args.render:
            env.render()

    cube_marker = grasp_marker = hit_marker = pre_marker = ref_marker = None
    if args.show_markers:
        cube_marker, _ = create_visual_marker(uw, "cube_center_marker", args.marker_radius, [1.0, 1.0, 0.0, 1.0])
        grasp_marker, _ = create_visual_marker(uw, "grasp_center_marker", args.marker_radius * 0.90, [0.1, 0.9, 1.0, 1.0])
        hit_marker, _ = create_visual_marker(uw, "hit_marker", args.marker_radius * 0.95, [1.0, 0.2, 1.0, 1.0])
        pre_marker, _ = create_visual_marker(uw, "pre_marker", args.marker_radius * 0.85, [0.2, 1.0, 0.2, 1.0])
        ref_marker, _ = create_visual_marker(uw, "ref_marker", args.marker_radius * 0.78, [1.0, 0.5, 0.0, 1.0])

    phase = "observe"
    observe_count = 0
    plan = None
    close_start_step = None
    post_start_step = None
    prev_cube_p = None
    prev_cube_yaw = None
    prev_grasp_center = None
    prev_front_yaw = None
    prev_ref_pos = None
    prev_ref_yaw = None
    locked_face_offset = None
    # Legacy finger-axis state is retained only as a diagnostic/fallback.
    prev_finger_yaw_wrapped = None
    finger_yaw_unwrapped = None

    # Robust hand-based control state.
    prev_hand_front_yaw_wrapped = None
    hand_front_yaw_unwrapped = None
    control_wz_filtered = None

    # Independent yaw-rate diagnostics:
    #   1) finger-axis yaw finite difference (existing front_wz),
    #   2) panda_hand quaternion-yaw finite difference,
    #   3) panda_hand direct world angular velocity from the simulator.
    prev_hand_yaw_wrapped = None
    hand_yaw_unwrapped = None

    post_hold_pos = None
    post_hold_yaw = None
    grasp_latched = False

    # y velocity-only controller state. Reset when entering/leaving dynamic phases.
    vy_int = 0.0

    close_schedule = build_close_schedule()
    last_phase = None
    printed_plan_step = None
    printed_descend_phase = False

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

    log_rows = []
    initial_cube_p = None
    initial_cube_yaw = None

    if physics_hits:
        print("[physics] applied:")
        for h in physics_hits:
            print(f"  - {h}")

    for t in range(args.steps):
        now_t = t * dt
        cube_p = np_pose(uw.cube.pose.p)
        cube_q = np_pose(uw.cube.pose.q)
        cube_yaw = float(quat_to_euler_xyz_wxyz(cube_q)[2])
        cube_v, cube_v_src = get_actor_linear_velocity(uw.cube, prev_p=prev_cube_p, dt=dt)
        cube_wz = 0.0 if prev_cube_yaw is None else wrap_to_pi(cube_yaw - prev_cube_yaw) / dt
        prev_cube_p = cube_p.copy()
        prev_cube_yaw = cube_yaw

        axis_point, axis_source_name, axis_source_kind, face_center, raw_mid, axis_open, finger_min_z = get_yaw_axis_point(
            uw,
            axis_z_offset=args.axis_z_offset,
            face_forward_offset=args.face_forward_offset,
            face_down_offset=args.face_down_offset,
        )

        # Use the gripper yaw/rotation-axis point only for planar predictive alignment.
        # get_yaw_axis_point() follows the old tracking script: prefer panda_hand / hand-like
        # link pose.p as the axis point, fallback to raw_mid if no hand-like link exists.
        # Keep individual finger positions for debug only, so we can still see early contact.
        lf_dbg, rf_dbg = find_finger_links(uw)
        left_finger_p = np_pose(lf_dbg.pose.p).astype(np.float32)
        right_finger_p = np_pose(rf_dbg.pose.p).astype(np.float32)
        finger_mid = (0.5 * (left_finger_p + right_finger_p)).astype(np.float32)
        finger_axis_yaw = wrap_to_pi(
            yaw_from_front_normal(axis_open) + math.radians(args.gripper_front_bias_deg)
        )
        if prev_finger_yaw_wrapped is None or finger_yaw_unwrapped is None:
            finger_yaw_unwrapped = float(finger_axis_yaw)
            finger_axis_wz = 0.0
        else:
            d_finger = wrap_to_pi(finger_axis_yaw - prev_finger_yaw_wrapped)
            finger_yaw_unwrapped = float(finger_yaw_unwrapped + d_finger)
            finger_axis_wz = float(d_finger / dt)
        prev_finger_yaw_wrapped = float(finger_axis_yaw)
        prev_front_yaw = finger_axis_yaw

        # ---- Independent hand-orientation diagnostics (logging only) ----
        # Use the panda_hand/link quaternion rather than the line joining the two
        # fingers.  Comparing the two finite-difference rates tells us whether a
        # spike comes from the finger-axis construction or from the hand pose too.
        hand_link_diag, hand_link_name_diag, _ = find_hand_like_link(uw)
        if hand_link_diag is not None:
            hand_q_diag = np_pose(hand_link_diag.pose.q)
            hand_rpy_diag = quat_to_euler_xyz_wxyz(hand_q_diag)
            hand_roll = float(hand_rpy_diag[0])
            hand_pitch = float(hand_rpy_diag[1])
            hand_yaw = float(hand_rpy_diag[2])
            if prev_hand_yaw_wrapped is None or hand_yaw_unwrapped is None:
                hand_yaw_unwrapped = float(hand_yaw)
                hand_yaw_wz_fd = 0.0
            else:
                d_hand_yaw = wrap_to_pi(hand_yaw - prev_hand_yaw_wrapped)
                hand_yaw_unwrapped = float(hand_yaw_unwrapped + d_hand_yaw)
                hand_yaw_wz_fd = float(d_hand_yaw / dt)
            prev_hand_yaw_wrapped = float(hand_yaw)
            hand_w_direct, hand_w_source = get_actor_angular_velocity(hand_link_diag)
        else:
            hand_roll = float("nan")
            hand_pitch = float("nan")
            hand_yaw = float("nan")
            hand_yaw_wz_fd = float("nan")
            hand_w_direct = np.full(3, np.nan, dtype=np.float32)
            hand_w_source = "no_hand_link"
            hand_link_name_diag = "none"

        # ---- Yaw state used by planning and control ----
        # Use a rigid panda_hand local axis directly. Do NOT calibrate it from the
        # line between finger link origins: in this model that vector is mostly Z,
        # so its tiny XY projection can point diagonally and inject a fixed yaw bias.
        use_hand_state = (
            args.yaw_state_source == "hand"
            and hand_link_diag is not None
            and math.isfinite(hand_yaw)
        )
        if use_hand_state:
            hand_front_yaw = projected_local_axis_yaw_wxyz(
                hand_q_diag,
                axis_index=0,
                bias_deg=float(args.gripper_front_bias_deg),
            )
            if (
                prev_hand_front_yaw_wrapped is None
                or hand_front_yaw_unwrapped is None
            ):
                hand_front_yaw_unwrapped = float(hand_front_yaw)
            else:
                d_hand_front = wrap_to_pi(
                    float(hand_front_yaw) - float(prev_hand_front_yaw_wrapped)
                )
                hand_front_yaw_unwrapped = float(
                    hand_front_yaw_unwrapped + d_hand_front
                )
            prev_hand_front_yaw_wrapped = float(hand_front_yaw)

            direct_wz = float(hand_w_direct[2])
            if not math.isfinite(direct_wz):
                direct_wz = float(hand_yaw_wz_fd)

            alpha = float(np.clip(args.wz_filter_alpha, 0.0, 1.0))
            if control_wz_filtered is None or not math.isfinite(control_wz_filtered):
                control_wz_filtered = direct_wz
            else:
                control_wz_filtered = (
                    alpha * direct_wz
                    + (1.0 - alpha) * float(control_wz_filtered)
                )

            front_yaw = float(hand_front_yaw)
            front_yaw_unwrapped = float(hand_front_yaw_unwrapped)
            front_wz = float(control_wz_filtered)
            control_yaw_source = "panda_hand"
        else:
            front_yaw = float(finger_axis_yaw)
            front_yaw_unwrapped = float(finger_yaw_unwrapped)
            front_wz = float(finger_axis_wz)
            control_yaw_source = "finger_axis"

        # Hybrid control point:
        #   - XY comes from the gripper yaw/rotation-axis point.
        #     This is the projected point we want to align with the object's rotation axis.
        #   - Z comes from the real left/right finger midpoint.
        #     This keeps the physical fingers at cube_z + wait_hover_clearance while waiting,
        #     instead of putting panda_hand/axis_point at that height and letting fingers dip
        #     into the object.
        # In short: predict/align axis_point in the XY plane, but control height with finger_mid.
        axis_point_actual = axis_point.copy()
        grasp_center = np.array([axis_point[0], axis_point[1], finger_mid[2]], dtype=np.float32)
        ee_v = np.zeros(3, dtype=np.float32) if prev_grasp_center is None else ((grasp_center - prev_grasp_center) / dt).astype(np.float32)
        prev_grasp_center = grasp_center.copy()

        if phase == "observe":
            observe_count += 1
            if observe_count >= args.observe_steps:
                phase = "plan"

        if phase == "plan":
            plan = choose_intercept_plan(
                step_idx=t,
                now_t=now_t,
                cube_p=cube_p,
                cube_v=cube_v,
                cube_yaw=cube_yaw,
                cube_wz=cube_wz,
                grasp_center=grasp_center,
                front_yaw_unwrapped=front_yaw_unwrapped,
                args=args,
                locked_face_offset=locked_face_offset,
            )
            close_start_step = None
            post_start_step = None
            post_hold_pos = None
            post_hold_yaw = None
            grasp_latched = False
            vy_int = 0.0
            if plan is None:
                phase = "observe"
                observe_count = 0
            else:
                phase = "go_wait"
                if printed_plan_step != plan.plan_step:
                    print_plan_debug(plan, grasp_center=grasp_center, cube_p=cube_p, cube_v=cube_v)
                    printed_plan_step = plan.plan_step
                    printed_descend_phase = False

        time_to_hit = None if plan is None else float(plan.hit_time - now_t)
        ref_pos = grasp_center.copy()
        ref_vel = np.zeros(3, dtype=np.float32)
        ref_yaw = front_yaw
        ref_wz = 0.0
        launch_s = 0.0

        if phase == "observe":
            ref_pos = grasp_center.copy()
            ref_pos[2] = max(float(grasp_center[2]), float(cube_p[2] + args.observe_hover_clearance))
            ref_vel = np.zeros(3, dtype=np.float32)

        if plan is not None:
            wait_err_xy = float(np.linalg.norm((plan.wait_pos[:2] - grasp_center[:2]).astype(np.float64)))
            wait_err_z = abs(float(plan.wait_pos[2] - grasp_center[2]))
            at_wait = (wait_err_xy <= args.go_wait_xy_eps and wait_err_z <= args.go_wait_z_eps)

            if phase not in ("post_grasp", "done"):
                if now_t >= plan.close_start_time:
                    phase = "close"
                    if close_start_step is None:
                        close_start_step = t
                elif now_t >= plan.descend_start_time:
                    phase = "descend"
                elif now_t >= plan.lin_start_time:
                    phase = "launch"
                elif at_wait:
                    phase = "wait_hold"
                else:
                    phase = "go_wait"

            if phase != last_phase:
                if phase in ("go_wait", "wait_hold", "launch", "descend", "close", "post_grasp"):
                    print_phase_debug(
                        tag=phase,
                        now_t=now_t,
                        plan=plan,
                        grasp_center=grasp_center,
                        ee_v=ee_v,
                        front_yaw=front_yaw,
                        front_wz=front_wz,
                    )
                    if phase == "launch":
                        printed_descend_phase = False
                        vy_int = 0.0
                    elif phase in ("go_wait", "wait_hold", "post_grasp"):
                        vy_int = 0.0
                last_phase = phase

            if phase in ("go_wait", "wait_hold"):
                ref_pos = plan.wait_pos.copy()
                ref_vel = np.zeros(3, dtype=np.float32)
            elif phase in ("launch", "descend"):
                launch_elapsed = float(np.clip(now_t - plan.lin_start_time, 0.0, plan.lin_total_time))

                ref_pos = plan.wait_pos.copy()
                ref_vel = np.zeros(3, dtype=np.float32)

                if args.variant == "predictive_intercept":
                    # Pure predictive interception: move from the waiting point to the
                    # predicted contact point with a smooth time law that has zero
                    # terminal XY velocity. This keeps the same predicted hit point and
                    # hit time, but intentionally does not synchronize target velocity.
                    intercept_T = max(1e-6, float(plan.hit_time - plan.lin_start_time))
                    intercept_tau = float(
                        np.clip(now_t - plan.lin_start_time, 0.0, intercept_T)
                    )
                    intercept_alpha = intercept_tau / intercept_T
                    s_xy = smoothstep01(intercept_alpha)
                    ds_xy = smoothstep01_derivative(intercept_alpha) / intercept_T
                    delta_xy = (
                        plan.hit_pos[:2] - plan.wait_pos[:2]
                    ).astype(np.float32)
                    ref_pos[:2] = plan.wait_pos[:2] + float(s_xy) * delta_xy
                    ref_vel[:2] = float(ds_xy) * delta_xy
                    launch_s = float(intercept_alpha)
                else:
                    # Linear/full synchronization: preserve the tuned profile that
                    # reaches the predicted contact point with target XY velocity.
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
                    launch_s = float(
                        np.clip(
                            launch_elapsed / max(1e-6, plan.lin_total_time),
                            0.0,
                            1.0,
                        )
                    )

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

        if phase == "close" and plan is not None:
            ref_pos = plan.close_pos.copy()
            if args.variant == "predictive_intercept":
                # Stop at the predicted interception point: no terminal velocity sync.
                ref_vel = np.zeros(3, dtype=np.float32)
            else:
                ref_vel = plan.target_vel.copy()
                ref_vel[2] = 0.0

        if phase == "post_grasp" and plan is not None:
            if post_hold_pos is None:
                post_hold_pos = grasp_center.copy()
                post_hold_yaw = float(front_yaw)
            ref_pos = post_hold_pos.copy()
            ref_vel = np.zeros(3, dtype=np.float32)
            if post_start_step is not None and (t - post_start_step) >= args.post_steps:
                phase = "done"

        # Three-stage ablation:
        #   predictive_intercept: predicted contact point/time, zero terminal XY
        #                         velocity, pre-align to hit face, ref_wz = 0.
        #   linear_sync:          target XY velocity sync, pre-align to hit face,
        #                         ref_wz = 0.
        #   full_sync:            target XY velocity sync plus target angular-rate sync.
        if args.enable_yaw_sync and plan is not None:
            if args.variant in ("predictive_intercept", "linear_sync"):
                if phase in ("go_wait", "wait_hold", "launch", "descend", "close"):
                    ref_yaw = float(plan.hit_yaw)
                    ref_wz = 0.0
                elif phase == "post_grasp" and post_hold_yaw is not None:
                    ref_yaw = float(post_hold_yaw)
                    ref_wz = 0.0
            else:
                if phase in ("go_wait", "wait_hold"):
                    ref_yaw = float(plan.wait_yaw)
                    ref_wz = float(plan.target_wz) if args.yaw_wait_wz else 0.0
                elif phase in ("launch", "descend", "close"):
                    ref_yaw = float(
                        plan.hit_yaw + plan.target_wz * (now_t - plan.hit_time)
                    )
                    ref_wz = float(plan.target_wz)
                elif phase == "post_grasp" and post_hold_yaw is not None:
                    ref_yaw = float(post_hold_yaw)
                    ref_wz = 0.0

        proj_z = float(cube_p[2] + args.marker_z_offset)
        set_marker_pose(cube_marker, [cube_p[0], cube_p[1], proj_z])
        set_marker_pose(grasp_marker, [grasp_center[0], grasp_center[1], proj_z])
        if plan is not None:
            set_marker_pose(hit_marker, [plan.hit_pos[0], plan.hit_pos[1], proj_z])
            set_marker_pose(pre_marker, [plan.wait_pos[0], plan.wait_pos[1], proj_z])
        set_marker_pose(ref_marker, [ref_pos[0], ref_pos[1], proj_z])

        action = np.zeros(7, dtype=np.float32)
        action[-1] = +1.0

        # Debug terms for y velocity-only controller.
        vy_only_active = False
        vy_err_only = 0.0
        vy_dbg_ff = 0.0
        vy_dbg_fb = 0.0
        vy_dbg_i = 0.0

        if phase in ("observe", "go_wait", "wait_hold", "launch", "descend", "close", "post_grasp"):
            kp_xyz, kv_xyz, ff_xyz = get_phase_gains(phase, args)

            u_xyz, pos_err_vec, vel_err_vec = control_xyz(
                ref_pos=ref_pos,
                ref_vel=ref_vel,
                cur_pos=grasp_center,
                cur_vel=ee_v,
                kp_xyz=kp_xyz,
                kv_xyz=kv_xyz,
                ff_xyz=ff_xyz,
                max_cmd=1.0,
            )
            axis_scale = get_axis_scale(phase, args)
            action[:3] = u_xyz * axis_scale

            # During descend, prevent the z controller from creeping down
            # too slowly when the reference is clearly below the current finger-mid height.
            # This does not change close timing or close gripper commands; it only helps
            # the fingers reach the already-planned close height before close starts.
            if phase == "descend" and float(pos_err_vec[2]) < -float(args.descend_z_far_eps):
                min_z = float(args.descend_z_min_action)
                if abs(float(action[2])) < min_z:
                    action[2] = -min_z

            # Fast go_wait: keep a minimum command while still clearly far from
            # wait_pos. This avoids the old slow exponential tail where action
            # became tiny before the gripper had really settled at the wait point.
            if phase == "go_wait" and args.fast_go_wait:
                action[:3] = np.clip(action[:3], -float(args.go_wait_action_clip), float(args.go_wait_action_clip))
                for _idx in (0, 1):
                    if abs(float(pos_err_vec[_idx])) > float(args.go_wait_far_xy):
                        min_cmd = float(args.go_wait_min_xy_action)
                        if abs(float(action[_idx])) < min_cmd:
                            action[_idx] = math.copysign(min_cmd, float(pos_err_vec[_idx]))
                if abs(float(pos_err_vec[2])) > float(args.go_wait_far_z):
                    min_cmd = float(args.go_wait_min_z_action)
                    if abs(float(action[2])) < min_cmd:
                        action[2] = math.copysign(min_cmd, float(pos_err_vec[2]))

            # In dynamic intercept phases, y is controlled by velocity only.
            # This removes y position error from action[1] completely.
            # x and z keep the original controller.
            if args.vy_vel_only and phase in ("launch", "descend", "close"):
                action_y, vy_int, vy_err_only, vy_dbg_ff, vy_dbg_fb, vy_dbg_i = control_y_velocity_only(
                    target_vy=float(ref_vel[1]),
                    cur_vy=float(ee_v[1]),
                    dt=float(dt),
                    vy_int=float(vy_int),
                    ee_v_per_action_y=float(args.ee_v_per_action_y),
                    kv_action=float(args.vy_kv_action),
                    ki_action=float(args.vy_ki_action),
                    int_clip=float(args.vy_int_clip),
                    action_clip=float(args.vy_action_clip),
                )
                action[1] = float(action_y)
                vy_only_active = True

            yaw_err = 0.0
            w_err = 0.0
            a5_ff = 0.0
            a5_fb = 0.0
            action[5] = 0.0
            if args.enable_yaw_sync and phase in ("go_wait", "wait_hold", "launch", "descend", "close"):
                # Use axis-periodic yaw error to avoid commanding an unnecessary >90 deg turn
                # for the symmetric gripper/cube-face alignment.
                yaw_err = axis_yaw_err(float(ref_yaw), float(front_yaw_unwrapped))
                w_err = float(ref_wz - front_wz)
                yaw_u = float(args.kp_yaw_sync) * yaw_err + float(args.kp_w_sync) * w_err
                a5_ff = float(args.yaw_ff_gain) * float(args.a5_per_front_wz_deg) * math.degrees(float(ref_wz))
                a5_fb = -yaw_u
                action[5] = float(np.clip(a5_ff + a5_fb, -float(args.yaw_clip), float(args.yaw_clip)))

            if phase == "close":
                close_step = 0 if close_start_step is None else max(0, t - close_start_step)
                idx = min(len(close_schedule) - 1, close_step // max(1, args.close_ramp_step))
                action[-1] = float(close_schedule[idx])
                if idx >= len(close_schedule) - 2:
                    phase = "post_grasp"
                    post_start_step = t
                    grasp_latched = True
            elif phase == "post_grasp":
                action[-1] = float(args.post_hold_grip)
            else:
                action[-1] = +1.0
        else:
            pos_err_vec = np.zeros(3, dtype=np.float32)
            vel_err_vec = np.zeros(3, dtype=np.float32)
            yaw_err = 0.0
            w_err = 0.0
            a5_ff = 0.0
            a5_fb = 0.0

        # Once the close ramp has reached the holding stage, keep the gripper closed
        # even after the finite-state machine enters done. Without this latch, the
        # default action[-1] = +1.0 opens the gripper in done and releases the cube.
        if grasp_latched or phase == "done":
            action[-1] = float(args.post_hold_grip)

        plan_tau = float(plan.tau) if plan is not None else float("nan")
        plan_hit = float(plan.hit_time) if plan is not None else float("nan")

        if initial_cube_p is None:
            initial_cube_p = cube_p.copy()
            initial_cube_yaw = float(cube_yaw)

        if args.log_csv and (t % max(1, int(args.log_every)) == 0 or t == args.steps - 1):
            rel_pos = (grasp_center - cube_p).astype(np.float32)
            rel_vel = (ee_v - cube_v).astype(np.float32)
            log_rows.append({
                "method": args.method_name,
                "variant": args.variant,
                "seed": int(args.seed),
                "t": int(t),
                "time": _safe_float(now_t),
                "phase": phase,
                "drift_speed": _safe_float(args.drift_speed),
                "spin_speed": _safe_float(args.spin_speed),
                "y_offset": _safe_float(args.y_offset),
                "cube_x": _safe_float(cube_p[0]),
                "cube_y": _safe_float(cube_p[1]),
                "cube_z": _safe_float(cube_p[2]),
                "cube_vx": _safe_float(cube_v[0]),
                "cube_vy": _safe_float(cube_v[1]),
                "cube_vz": _safe_float(cube_v[2]),
                "cube_yaw": _safe_float(cube_yaw),
                "cube_yaw_deg": _safe_float(math.degrees(cube_yaw)),
                "cube_wz": _safe_float(cube_wz),
                "cube_wz_deg": _safe_float(math.degrees(cube_wz)),
                "grasp_x": _safe_float(grasp_center[0]),
                "grasp_y": _safe_float(grasp_center[1]),
                "grasp_z": _safe_float(grasp_center[2]),
                "ee_vx": _safe_float(ee_v[0]),
                "ee_vy": _safe_float(ee_v[1]),
                "ee_vz": _safe_float(ee_v[2]),
                "front_yaw": _safe_float(front_yaw),
                "front_yaw_deg": _safe_float(math.degrees(front_yaw)),
                "front_yaw_unwrapped": _safe_float(front_yaw_unwrapped),
                "front_wz": _safe_float(front_wz),
                "front_wz_deg": _safe_float(math.degrees(front_wz)),
                "control_yaw_source": str(control_yaw_source),
                "hand_axis_definition": "panda_hand_local_x_projected" if use_hand_state else "legacy_finger_projection",
                "control_front_yaw": _safe_float(front_yaw),
                "control_front_yaw_deg": _safe_float(math.degrees(front_yaw)),
                "control_front_wz": _safe_float(front_wz),
                "control_front_wz_deg": _safe_float(math.degrees(front_wz)),
                # Three-way yaw-rate diagnostic.  The legacy finger state is retained
                # unchanged for backward compatibility and controller behavior.
                "finger_axis_yaw": _safe_float(finger_axis_yaw),
                "finger_axis_yaw_deg": _safe_float(math.degrees(finger_axis_yaw)),
                "finger_axis_wz_fd": _safe_float(finger_axis_wz),
                "finger_axis_wz_fd_deg": _safe_float(math.degrees(finger_axis_wz)),
                "hand_roll": _safe_float(hand_roll),
                "hand_roll_deg": _safe_float(math.degrees(hand_roll)),
                "hand_pitch": _safe_float(hand_pitch),
                "hand_pitch_deg": _safe_float(math.degrees(hand_pitch)),
                "hand_yaw": _safe_float(hand_yaw),
                "hand_yaw_deg": _safe_float(math.degrees(hand_yaw)),
                "hand_yaw_unwrapped": _safe_float(hand_yaw_unwrapped),
                "hand_yaw_wz_fd": _safe_float(hand_yaw_wz_fd),
                "hand_yaw_wz_fd_deg": _safe_float(math.degrees(hand_yaw_wz_fd)),
                "hand_wx_direct": _safe_float(hand_w_direct[0]),
                "hand_wy_direct": _safe_float(hand_w_direct[1]),
                "hand_wz_direct": _safe_float(hand_w_direct[2]),
                "hand_wx_direct_deg": _safe_float(math.degrees(hand_w_direct[0])),
                "hand_wy_direct_deg": _safe_float(math.degrees(hand_w_direct[1])),
                "hand_wz_direct_deg": _safe_float(math.degrees(hand_w_direct[2])),
                "hand_w_source": str(hand_w_source),
                "hand_link_name": str(hand_link_name_diag),

                "ref_x": _safe_float(ref_pos[0]),
                "ref_y": _safe_float(ref_pos[1]),
                "ref_z": _safe_float(ref_pos[2]),
                "ref_vx": _safe_float(ref_vel[0]),
                "ref_vy": _safe_float(ref_vel[1]),
                "ref_vz": _safe_float(ref_vel[2]),
                "ref_yaw": _safe_float(ref_yaw),
                "ref_yaw_deg": _safe_float(math.degrees(ref_yaw)),
                "ref_wz": _safe_float(ref_wz),
                "ref_wz_deg": _safe_float(math.degrees(ref_wz)),
                "rel_x": _safe_float(rel_pos[0]),
                "rel_y": _safe_float(rel_pos[1]),
                "rel_z": _safe_float(rel_pos[2]),
                "rel_vx": _safe_float(rel_vel[0]),
                "rel_vy": _safe_float(rel_vel[1]),
                "rel_vz": _safe_float(rel_vel[2]),
                "rel_speed": _safe_float(np.linalg.norm(rel_vel)),
                "pos_err_x": _safe_float(pos_err_vec[0]),
                "pos_err_y": _safe_float(pos_err_vec[1]),
                "pos_err_z": _safe_float(pos_err_vec[2]),
                "pos_err_norm": _safe_float(np.linalg.norm(pos_err_vec)),
                "vel_err_x": _safe_float(vel_err_vec[0]),
                "vel_err_y": _safe_float(vel_err_vec[1]),
                "vel_err_z": _safe_float(vel_err_vec[2]),
                "vel_err_norm": _safe_float(np.linalg.norm(vel_err_vec)),
                "yaw_err": _safe_float(yaw_err),
                "yaw_err_deg": _safe_float(math.degrees(yaw_err)),
                "w_err": _safe_float(w_err),
                "w_err_deg": _safe_float(math.degrees(w_err)),
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
            })

        if args.keep_awake_every_step:
            try_wake(uw.cube)
        env.step(action)
        if args.keep_awake_every_step:
            try_wake(uw.cube)
        if args.render:
            env.render()

        plan_tau = float(plan.tau) if plan is not None else float("nan")
        plan_hit = float(plan.hit_time) if plan is not None else float("nan")
        if t % args.debug_every == 0 or t == args.steps - 1:
            print(
                f"[t={t:03d}] phase={phase:<11s} "
                f"cube=({cube_p[0]:+.3f},{cube_p[1]:+.3f},{cube_p[2]:+.3f}) "
                f"ctrl_pt=({grasp_center[0]:+.3f},{grasp_center[1]:+.3f},{grasp_center[2]:+.3f}) "
                f"axis_pt=({axis_point_actual[0]:+.3f},{axis_point_actual[1]:+.3f},{axis_point_actual[2]:+.3f}) "
                f"finger_mid=({finger_mid[0]:+.3f},{finger_mid[1]:+.3f},{finger_mid[2]:+.3f}) "
                f"Lrel=({left_finger_p[0]-cube_p[0]:+.3f},{left_finger_p[1]-cube_p[1]:+.3f},{left_finger_p[2]-cube_p[2]:+.3f}) "
                f"Rrel=({right_finger_p[0]-cube_p[0]:+.3f},{right_finger_p[1]-cube_p[1]:+.3f},{right_finger_p[2]-cube_p[2]:+.3f}) "
                f"ctrl_rel=({grasp_center[0]-cube_p[0]:+.3f},{grasp_center[1]-cube_p[1]:+.3f},{grasp_center[2]-cube_p[2]:+.3f}) "
                f"axis_rel=({axis_point_actual[0]-cube_p[0]:+.3f},{axis_point_actual[1]-cube_p[1]:+.3f},{axis_point_actual[2]-cube_p[2]:+.3f}) "
                f"mid_rel=({finger_mid[0]-cube_p[0]:+.3f},{finger_mid[1]-cube_p[1]:+.3f},{finger_mid[2]-cube_p[2]:+.3f}) "
                f"ref=({ref_pos[0]:+.3f},{ref_pos[1]:+.3f},{ref_pos[2]:+.3f}) "
                f"cube_v=({cube_v[0]:+.3f},{cube_v[1]:+.3f},{cube_v[2]:+.3f}) ee_v=({ee_v[0]:+.3f},{ee_v[1]:+.3f},{ee_v[2]:+.3f}) "
                f"ref_v=({ref_vel[0]:+.3f},{ref_vel[1]:+.3f},{ref_vel[2]:+.3f}) "
                f"front_yaw={math.degrees(front_yaw):+.2f} ref_yaw={math.degrees(ref_yaw):+.2f} yaw_err={math.degrees(yaw_err):+.2f} "
                f"cube_wz={math.degrees(cube_wz):+.2f} ref_wz={math.degrees(ref_wz):+.2f} werr={math.degrees(w_err):+.2f} "
                f"pos_err=({pos_err_vec[0]:+.3f},{pos_err_vec[1]:+.3f},{pos_err_vec[2]:+.3f}) vel_err=({vel_err_vec[0]:+.3f},{vel_err_vec[1]:+.3f},{vel_err_vec[2]:+.3f}) "
                f"plan_tau={plan_tau:+.3f} hit_t={plan_hit:+.3f} t_hit={time_to_hit if time_to_hit is not None else float('nan'):+.3f} launch_s={launch_s:+.2f} "
                f"a_xyz=({action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f}) a5={action[5]:+.3f} grip={action[-1]:+.2f} "
                f"vy_only={int(vy_only_active)} vy_err={vy_err_only:+.3f} "
                f"vy_ctrl=(ff={vy_dbg_ff:+.3f},fb={vy_dbg_fb:+.3f},i={vy_dbg_i:+.3f},int={vy_int:+.3f}) "
                f"cube_v_src={cube_v_src} axis_src={axis_source_name}"
            )
    # Write experiment logs and one-line summary. This is intentionally outside the
    # controller loop so it cannot affect the control logic.
    if args.log_csv and log_rows:
        _write_rows_csv(args.log_csv, log_rows)
        print(f"[log] wrote {len(log_rows)} rows to {args.log_csv}")

    if args.summary_csv:
        final_cube_p = cube_p.copy() if 'cube_p' in locals() else np.zeros(3, dtype=np.float32)
        final_cube_yaw = float(cube_yaw) if 'cube_yaw' in locals() else float('nan')
        if initial_cube_p is None:
            initial_cube_p = final_cube_p.copy()
        if initial_cube_yaw is None:
            initial_cube_yaw = final_cube_yaw
        cube_disp = final_cube_p - initial_cube_p
        first_close_row = _first_row(log_rows, lambda r: str(r.get("phase")) in ("close", "post_grasp", "done") or float(r.get("grip_cmd", 1.0)) < 0.0)
        first_grip_row = _first_row(log_rows, lambda r: float(r.get("grip_cmd", 1.0)) < 0.0)
        final_row = log_rows[-1] if log_rows else {}
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
            "entered_close": int(first_close_row is not None),
            "first_close_t": _safe_float(first_close_row.get("time", float("nan"))) if first_close_row else float("nan"),
            "first_grip_t": _safe_float(first_grip_row.get("time", float("nan"))) if first_grip_row else float("nan"),
            "first_close_rel_speed": _safe_float(first_close_row.get("rel_speed", float("nan"))) if first_close_row else float("nan"),
            "first_grip_rel_speed": _safe_float(first_grip_row.get("rel_speed", float("nan"))) if first_grip_row else float("nan"),
            "first_close_yaw_err_deg": _safe_float(first_close_row.get("yaw_err_deg", float("nan"))) if first_close_row else float("nan"),
            "first_close_werr_deg": _safe_float(first_close_row.get("w_err_deg", float("nan"))) if first_close_row else float("nan"),
            "max_abs_yaw_err_deg": _max_abs(log_rows, "yaw_err_deg"),
            "max_abs_werr_deg": _max_abs(log_rows, "w_err_deg"),
            "max_pos_err_norm": max([_safe_float(r.get("pos_err_norm", float("nan"))) for r in log_rows] or [float("nan")]),
            "max_rel_speed": max([_safe_float(r.get("rel_speed", float("nan"))) for r in log_rows] or [float("nan")]),
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

    print(
        f"\n[summary] finished variant={args.variant} final_phase={phase} "
        f"grasp_latched={int(grasp_latched)} yaw_source={args.yaw_state_source} "
        f"axis_source={axis_source_name}"
    )
    env.close()


if __name__ == "__main__":
    main()

