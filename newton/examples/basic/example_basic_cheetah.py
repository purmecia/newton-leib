# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Cheetah
#
# Loads multiple half-cheetah robots from MJCF and lets them move under
# random joint torques using direct force control (matching dflex
# conventions).
#
# Uses half_cheetah_separate_root.xml which splits the three MJCF root
# joints (rootx, rootz, rooty) into separate bodies. This matches dflex's
# kinematic structure (9 joints, 9 bodies per cheetah including 2
# zero-mass intermediates) for simulation stability.
#
# Command: python -m newton.examples basic_cheetah --world-count 8
#
###########################################################################

import warp as wp

import newton
import newton.examples


@wp.kernel
def randomize_joint_torques_kernel(
    joint_f: wp.array(dtype=float),
    dofs_per_robot: int,
    root_dofs: int,
    action_strength: float,
    seed: int,
):
    """Apply random torques to actuated joints (direct force control)."""
    tid = wp.tid()
    rng = wp.rand_init(seed, tid)
    robot_idx = tid // (dofs_per_robot - root_dofs)
    local_act = tid % (dofs_per_robot - root_dofs)
    global_dof = robot_idx * dofs_per_robot + root_dofs + local_act
    action = wp.clamp(1.0 - 2.0 * wp.randf(rng), -1.0, 1.0)
    joint_f[global_dof] = action * action_strength


@wp.kernel
def _set_dflex_joint_params_kernel(
    joint_target_kd: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    dofs_per_robot: int,
    root_dofs: int,
):
    """Set per-DOF joint parameters to match dflex conventions."""
    tid = wp.tid()
    local_dof = tid % dofs_per_robot
    joint_limit_ke[tid] = 1000.0
    joint_limit_kd[tid] = 10.0
    # if local_dof < root_dofs:
    #     joint_target_kd[tid] = 0.0
    # else:
    #     joint_target_kd[tid] = 1.0


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.viewer = viewer
        self.step_count = 0

        self.dofs_per_robot = 9
        self.root_dofs = 3
        self.num_actuated = (self.dofs_per_robot - self.root_dofs) * self.world_count
        self.action_strength = 200.0

        cheetah = newton.ModelBuilder()
        cheetah.default_shape_cfg.ke = 2.0e4
        cheetah.default_shape_cfg.kd = 1.0e3
        cheetah.default_shape_cfg.kf = 1.0e3
        cheetah.default_shape_cfg.mu = 1.0
        cheetah.default_joint_cfg.target_ke = 0.0
        cheetah.default_joint_cfg.target_kd = 0.0
        cheetah.default_joint_cfg.armature = 0.1

        cheetah.add_mjcf(
            newton.examples.get_asset("half_cheetah_separate_root.xml"),
            ignore_names=["ground"],
        )

        scene = newton.ModelBuilder()
        scene.replicate(cheetah, self.world_count)
        scene.add_ground_plane(cfg=cheetah.default_shape_cfg)

        self.model = scene.finalize()

        total_dofs = self.dofs_per_robot * self.world_count
        wp.launch(
            _set_dflex_joint_params_kernel,
            dim=total_dofs,
            inputs=[
                self.model.joint_target_kd,
                self.model.joint_limit_ke,
                self.model.joint_limit_kd,
                self.dofs_per_robot,
                self.root_dofs,
            ],
        )

        self.solver = newton.solvers.SolverFeatherstone(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            try:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph
            except Exception as exc:
                self.graph = None
                wp.utils.warn(f"CUDA graph capture failed: {exc}")
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        wp.launch(
            randomize_joint_torques_kernel,
            dim=self.num_actuated,
            inputs=[self.control.joint_f, self.dofs_per_robot, self.root_dofs, self.action_strength, self.step_count],
        )

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self.step_count += 1

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cheetah bodies above ground",
            lambda q, qd: q[2] > -1.0,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=8)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
