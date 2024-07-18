# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class CassieRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 45
        num_actions = 12

    class commands( LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 2.
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-2., 2.] # min max [m/s] seems like less than or equal to 0.2 it sends 0 command
            lin_vel_y = [-0., 0.]   # min max [m/s]
            ang_vel_yaw = [-.5, .5]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class terrain( LeggedRobotCfg.terrain):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.001 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4] # 1mx1m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        difficulty = [0.2, 0.4, 0.6]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'hip_abduction_left': 0.1,
            'hip_rotation_left': 0.,
            'hip_flexion_left': 1.,
            'thigh_joint_left': -1.8,
            'ankle_joint_left': 1.57,
            'toe_joint_left': -1.57,

            'hip_abduction_right': -0.1,
            'hip_rotation_right': 0.,
            'hip_flexion_right': 1.,
            'thigh_joint_right': -1.8,
            'ankle_joint_right': 1.57,
            'toe_joint_right': -1.57
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'T'
        # PD Drive parameters:
        stiffness = {   'hip_abduction': 100.0, 'hip_rotation': 100.0,
                        'hip_flexion': 200., 'thigh_joint': 200., 'ankle_joint': 200.,
                        'toe_joint': 40.}  # [N*m/rad]
        damping = { 'hip_abduction': 3.0, 'hip_rotation': 3.0,
                    'hip_flexion': 6., 'thigh_joint': 6., 'ankle_joint': 6.,
                    'toe_joint': 1.}  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 30.
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cassie/urdf/cassie.urdf'
        name = "cassie"
        foot_name = 'toe'
        terminate_after_contacts_on = ['pelvis']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_lin_vel = 10.0
            energy_minimization = 4.0
            tracking_ang_vel = 1.0#1.0
            torques = 0.#-5.e-6
            dof_acc = 0.#-2.e-7
            lin_vel_z = 0.#-0.5
            feet_air_time = 0.#5.
            dof_pos_limits = 0.#-1.
            no_fly = 0.5#0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -1.e-2
            action_rate = 0.

class CassieRoughCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'OnPolicyRunnerHistory'

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO_sym'
        run_name = ''
        experiment_name = 'cassie'
        num_steps_per_env = 24 # per iteration
        max_iterations = 2000 # number of policy updates
        
        # Optional. Choose the length of state history for the algorithm to use.
        history_len = 15
        critic_history_len = 1

        # logging
        save_interval = 1000 # check for potential saves every this many iterations
        experiment_name = 'cassie'
        run_name = 'cassie'
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
        # symmetric loss


    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        
        
        # Symmetric loss
        mirror = {'hip_flexion': (2,8),
            'thigh_joint': (3,9),
            'ankle_joint': (4,10),
            'toe_joint': (5,11),
                        } # Joint pairs that need to be mirrored
        mirror_neg = {'HipYaw': (1,7), 'HipRoll': (0,6)} # Joint pairs that need to be mirrored and signs must be changed
        mirror_weight = 4
        # The following lists indicate the ranges in the observation vector indices, for which specific mirroring method should applied
        # For example, cartesian_angular_mirror = [(0,3), (6,12)] indicate that the cartesian angular mirror operation should be applied
        # to the 0th~2nd, and the 6th~8th, 9th~11th elements of the observation vector.
        cartesian_angular_mirror = [(0,3)]
        cartesian_linear_mirror = [(3,6)]
        cartesian_command_mirror = [(6,9)]
        # The following list indicate the ranges in the observation vector indices, for which switching places is necessary
        switch_mirror = []
        # The following list indicate the ranges in the observation vector indices, for which no mirroring is necessary.
        no_mirror = []


  