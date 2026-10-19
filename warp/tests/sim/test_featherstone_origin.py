# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FeatherstoneIntegrator far from the world origin.

In world-origin coordinates, spatial inertias grow with m |p|^2 and the mass matrix loses float32 precision; a floating
base, whose velocity is a twist about the world origin, also makes it ill-conditioned. The integrator works in a frame
at each articulation's root; these tests check that the dynamics and their gradients do not depend on where the scene
is placed.
"""

import unittest

import numpy as np

import warp as wp
import warp.sim
from warp.tests.unittest_utils import *

FAR = np.array([300.0, 0.0, 300.0])  # horizontal (the ground plane is y = 0); the unchanged integrator fails at 100 m
DT = 2.0e-3
STEPS = 150  # lands at about step 80; later the bouncing amplifies roundoff chaotically
ROOT_START = np.array([0.0, 0.25, 0.0])
ROOT_QD = np.array([0.5, -1.0, 1.0, 0.3, 0.0, -0.2])  # angular velocity, then velocity of the root body origin


@wp.kernel
def eval_loss(joint_q: wp.array(dtype=float), joint_qd: wp.array(dtype=float), loss: wp.array(dtype=float)):
    # Translation invariant: root height, child joint coordinates and velocities.
    s = joint_q[1]
    for i in range(7, joint_q.shape[0]):
        s += joint_q[i]
    for i in range(6, joint_qd.shape[0]):
        s += joint_qd[i]
    loss[0] = s


def twist_about_origin(root_qd: np.ndarray, p: np.ndarray) -> np.ndarray:
    """(omega, velocity of the point p) -> (omega, velocity of the point at the world origin)."""
    w, v = root_qd[:3], root_qd[3:]
    return np.concatenate([w, v - np.cross(w, p)])


def build_model(device, offset=None, free_armature=0.0, root_parent_xform=None, ground=True):
    """A free box with a ball-jointed child and a revolute grandchild, dropped onto the ground while spinning, next to
    a fixed-base revolute pendulum. Everything is shifted by `offset`; with `root_parent_xform` the root's joint frame
    is moved instead of its coordinates (same motion)."""
    offset = np.zeros(3) if offset is None else offset
    b = wp.sim.ModelBuilder()
    b.default_shape_ke = 1.0e4
    b.default_shape_kd = 1.0e2
    b.default_shape_kf = 1.0e2
    rot = wp.quat_from_axis_angle(wp.vec3(0.3, 1.0, 0.2), 0.4)
    root_pos = offset + ROOT_START

    root = b.add_body()
    b.add_shape_box(root, hx=0.2, hy=0.05, hz=0.1, density=1000.0)
    b.add_shape_sphere(root, pos=wp.vec3(0.0, -0.05, 0.0), radius=0.06, density=500.0)
    link1 = b.add_body()
    b.add_shape_capsule(link1, radius=0.03, half_height=0.1, up_axis=0, density=1000.0)
    link2 = b.add_body()
    b.add_shape_box(link2, hx=0.08, hy=0.02, hz=0.02, density=1000.0)

    if root_parent_xform is None:
        b.add_joint_free(root, armature=free_armature)
        X_root = wp.transform(wp.vec3(*root_pos), rot)
        qd_frame = wp.transform_identity()
    else:
        b.add_joint_free(root, parent_xform=root_parent_xform, armature=free_armature)
        X_root = wp.transform_inverse(root_parent_xform) * wp.transform(wp.vec3(*root_pos), rot)
        qd_frame = root_parent_xform
    b.add_joint_ball(
        root,
        link1,
        parent_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity()),
        armature=0.01,
    )
    b.add_joint_revolute(
        link1,
        link2,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
        armature=0.01,
    )

    pend = b.add_body()
    b.add_shape_box(pend, hx=0.05, hy=0.2, hz=0.05, density=1000.0)
    b.add_joint_revolute(
        -1,
        pend,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(*(offset + np.array([1.0, 1.0, 0.0]))), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.2, 0.0), wp.quat_identity()),
        armature=0.01,
    )

    b.joint_q[0:7] = [*wp.transform_get_translation(X_root), *wp.transform_get_rotation(X_root)]
    b.joint_q[7:11] = [*wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 1.0), 0.3)]
    b.joint_q[11] = 0.2
    b.joint_q[12] = 0.5
    # joint_qd of a free joint is a twist about the origin of its joint frame, expressed in that frame
    twist_world = twist_about_origin(ROOT_QD, root_pos)
    w_world, v_world = twist_world[:3], twist_world[3:]
    X_inv = wp.transform_inverse(qd_frame)
    q_inv, p_inv = wp.transform_get_rotation(X_inv), wp.transform_get_translation(X_inv)
    w = wp.quat_rotate(q_inv, wp.vec3(*w_world))
    v = wp.quat_rotate(q_inv, wp.vec3(*v_world)) + wp.cross(p_inv, w)
    b.joint_qd[0:6] = [*w, *v]
    b.joint_qd[6:9] = [0.5, -1.0, 0.3]
    b.joint_qd[9] = -1.0
    b.joint_qd[10] = 2.0

    model = b.finalize(device=device, requires_grad=True)
    model.ground = ground
    return model


def simulate(model, steps=STEPS, joint_act=None):
    """Returns the states and the loss of the final state, recorded on a tape."""
    integrator = wp.sim.FeatherstoneIntegrator(model, update_mass_matrix_every=4)
    control = model.control()
    if joint_act is not None:
        control.joint_act = joint_act
    states = [model.state(requires_grad=True) for _ in range(steps + 1)]
    loss = wp.zeros(1, dtype=float, device=model.device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, states[0])
        for i in range(steps):
            states[i].clear_forces()
            wp.sim.collide(model, states[i])
            integrator.simulate(model, states[i], states[i + 1], DT, control)
        wp.launch(
            eval_loss, dim=1, inputs=[states[-1].joint_q, states[-1].joint_qd], outputs=[loss], device=model.device
        )
    return states, loss, tape, control


def invariant_coordinates(state, offset):
    """Joint coordinates and velocities with the root position and twist expressed relative to `offset`."""
    q = state.joint_q.numpy().astype(np.float64)
    qd = state.joint_qd.numpy().astype(np.float64)
    q[0:3] -= offset
    qd[3:6] += np.cross(qd[0:3], offset)  # twist about the world origin -> twist about the offset
    return q, qd


def test_translation_invariance(test, device, free_armature=0.0):
    near_model = build_model(device, free_armature=free_armature)
    far_model = build_model(device, offset=FAR, free_armature=free_armature)
    near_states, near_loss, near_tape, near_control = simulate(near_model)
    far_states, far_loss, far_tape, far_control = simulate(far_model)

    for t in (STEPS // 3, STEPS):
        q0, qd0 = invariant_coordinates(near_states[t], np.zeros(3))
        q1, qd1 = invariant_coordinates(far_states[t], FAR)
        test.assertTrue(np.isfinite(q1).all() and np.isfinite(qd1).all())
        # float32 joint_q and joint_qd (a twist about the world origin) lose resolution in proportion to the offset;
        # measured at FAR: 4e-4 and 7e-3
        assert_np_equal(q1, q0, tol=2e-3)
        assert_np_equal(qd1, qd0, tol=3e-2)

    near_tape.backward(near_loss)
    far_tape.backward(far_loss)
    # joint_q: translation-invariant coordinates only. Measured relative errors: 5e-5 (actions), 3e-3 (joint_q).
    for g0, g1 in (
        (near_control.joint_act.grad.numpy(), far_control.joint_act.grad.numpy()),
        (near_states[0].joint_q.grad.numpy()[7:], far_states[0].joint_q.grad.numpy()[7:]),
    ):
        test.assertTrue(np.isfinite(g1).all())
        test.assertLess(np.linalg.norm(g1 - g0), 1e-2 * np.linalg.norm(g0))


def test_translation_invariance_free_armature(test, device):
    test_translation_invariance(test, device, free_armature=0.1)


def test_root_parent_xform(test, device):
    """Moving the root's joint frame instead of its coordinates does not change the motion."""
    X = wp.transform(wp.vec3(0.3, 0.1, -0.2), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5))
    plain, _, _, _ = simulate(build_model(device), steps=100)
    moved, _, _, _ = simulate(build_model(device, root_parent_xform=X), steps=100)
    assert_np_equal(moved[-1].body_q.numpy(), plain[-1].body_q.numpy(), tol=1e-3)


def test_gradient_finite_differences(test, device):
    """Tape gradient of the loss w.r.t. the initial child joint velocities and the joint actions, without contacts."""
    steps, eps = 30, 1e-2
    model = build_model(device, ground=False)
    states, loss, tape, control = simulate(model, steps=steps)
    tape.backward(loss)
    grads = {"joint_qd": states[0].joint_qd.grad.numpy(), "joint_act": control.joint_act.grad.numpy()}

    for name in grads:
        x0 = getattr(model, name).numpy() if name == "joint_qd" else control.joint_act.numpy()
        for i in range(6 if name == "joint_qd" else 0, len(x0)):
            losses = []
            for sign in (1.0, -1.0):
                x = x0.copy()
                x[i] += sign * eps
                m = build_model(device, ground=False)
                act = None
                if name == "joint_qd":
                    m.joint_qd.assign(x)
                else:
                    act = wp.array(x, dtype=float, device=device, requires_grad=True)
                losses.append(simulate(m, steps=steps, joint_act=act)[1].numpy()[0])
            fd = (losses[0] - losses[1]) / (2.0 * eps)
            test.assertAlmostEqual(grads[name][i], fd, delta=2e-2 * max(1.0, abs(fd)), msg=f"{name}[{i}]")


devices = get_test_devices()


class TestFeatherstoneOrigin(unittest.TestCase):
    pass


add_function_test(TestFeatherstoneOrigin, "test_translation_invariance", test_translation_invariance, devices=devices)
add_function_test(
    TestFeatherstoneOrigin,
    "test_translation_invariance_free_armature",
    test_translation_invariance_free_armature,
    devices=devices,
)
add_function_test(TestFeatherstoneOrigin, "test_root_parent_xform", test_root_parent_xform, devices=devices)
add_function_test(
    TestFeatherstoneOrigin, "test_gradient_finite_differences", test_gradient_finite_differences, devices=devices
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
