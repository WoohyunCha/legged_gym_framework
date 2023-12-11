from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.base.base_config import BaseConfig
from math import pi

class JetFlatCfg_fixed( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48 # 169 + 16*3
        num_actions = 12 # 12(lower) + 16(upper)

    
    class terrain( LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class init_state( LeggedRobotCfg.init_state ):
        deg2rad = pi / 180
        pos = [0.0, 0.0, .85] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "L_HipYaw" : 0.,
            "L_HipRoll" : 0.034906585,
            "L_HipPitch" : -0.034906585,
            "L_KneePitch" : 0.733038285,
            "L_AnklePitch" : -0.6981317,
            "L_AnkleRoll" : -0.034906585,
            "R_HipYaw" : 0,
            "R_HipRoll" : -0.034906585,
            "R_HipPitch" : 0.034906585,
            "R_KneePitch" : -0.733038285,
            "R_AnklePitch" : 0.6981317,
            "R_AnkleRoll" : 0.034906585
            ,
            "WaistPitch" : 0.,
            "WaistYaw" : 0.,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {   'HipYaw': 600., 'HipRoll': 200., 'HipPitch': 200., 
                        'KneePitch': 600., 
                        'AnklePitch': 600.,'AnkleRoll': 200.,
                        #'WaistPitch': 2000., 'WasitYaw': 2000.,
                        #'ShoulderPitch': 2000., 'ShoulderRoll': 2000., 'ShoulderYaw': 2000.,
                        #'ElbowRoll': 2000., 

                        }  # [N*m/rad]
        damping = { 'HipYaw': 10., 'HipRoll': 10., 'HipPitch': 10., 
                        'KneePitch': 10., 
                        'AnklePitch': 10.,'AnkleRoll': 10.,
                        #'WaistPitch': 100., 'WasitYaw': 100.,
                        #'ShoulderPitch': 100., 'ShoulderRoll': 100., 'ShoulderYaw': 100.,
                        #'ElbowRoll': 100., 

                        }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/jet/urdf/jet_fixed.urdf'
        name = "jet_fixed"
        foot_name = 'AnkleRoll'
        terminate_after_contacts_on = ['Hip', 'Waist', 'Knee', 'base'] # check legged_robot.py->_create_envs->uses links of which names contain "terminate_after_contacts_on"
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 1000.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -300.
            tracking_ang_vel = 1.0
            torques = -5.e-6
            dof_acc = -2.e-7 # -2.e-7
            lin_vel_z = -0.5
            feet_air_time = .5 # 5.
            dof_pos_limits = -5.    # -1.
            no_fly = 0.25 # .25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -1.e-3 # -1.e-4
            tracking_lin_vel = 2.0
            angular_momentum = 1.

    class commands:
        curriculum = True
        max_curriculum = 4.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
            

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'sigmoid' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
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

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
        
class JetFlatCfgPPO_fixed( LeggedRobotCfgPPO):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_jet_fixed'

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

