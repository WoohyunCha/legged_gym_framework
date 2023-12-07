from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.base.base_config import BaseConfig
from math import pi

class JetRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 2048
        num_observations = 217 # 169 + 16*3
        num_actions = 28 # 12(lower) + 16(upper)

    
    class terrain( LeggedRobotCfg.terrain):
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    class init_state( LeggedRobotCfg.init_state ):
        deg2rad = pi / 180
        pos = [0.0, 0.0, 0.718776] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "L_HipYaw" : 0.,
            "L_HipRoll" : 0.,
            "L_HipPitch" : 24.0799945102 * deg2rad,
            "L_KneePitch" : -14.8197729791 * deg2rad,
            "L_AnklePitch" : -9.2602215311 * deg2rad,
            "L_AnkleRoll" : 0.,
            "R_HipYaw" : 0.,
            "R_HipRoll" : 0.,
            "R_HipPitch" : -24.0799945102 * deg2rad,
            "R_KneePitch" : 14.8197729791 * deg2rad,
            "R_AnklePitch" : 9.2602215311 * deg2rad,
            "R_AnkleRoll" : 0.
            ,
            "WaistPitch" : 0.,
            "WaistYaw" : 0.,
            "L_ShoulderPitch" : 0.698191,
            "L_ShoulderRoll" : -1.65828,
            "L_ShoulderYaw" : -1.39608,
            "L_ElbowRoll" : -1.91976,
            "L_WristYaw" : 0.,
            "L_WristRoll" : -1.22173,
            "L_HandYaw" : -0.174533,
            "R_ShoulderPitch" : -0.697935,
            "R_ShoulderRoll" : 1.65848,
            "R_ShoulderYaw" : 1.39608,
            "R_ElbowRoll" : 1.91976,
            "R_WristYaw" : 0,
            "R_WristRoll" : 1.22173,
            "R_HandYaw" : 0.174533,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {   'HipYaw': 2000., 'HipRoll': 1000., 'HipPitch': 1000., 
                        'KneePitch': 2000., 
                        'AnklePitch': 2000.,'AnkleRoll': 2000.,
                        'WaistPitch': 2000., 'WasitYaw': 2000.,
                        'ShoulderPitch': 2000., 'ShoulderRoll': 2000., 'ShoulderYaw': 2000.,
                        'ElbowRoll': 2000., 
                        'WristYaw': 2000., 'WristRoll': 2000.,
                        'HandYaw': 2000.,
                        }  # [N*m/rad]
        damping = { 'HipYaw': 600., 'HipRoll': 600., 'HipPitch': 600., 
                        'KneePitch': 600., 
                        'AnklePitch': 600.,'AnkleRoll': 600.,
                        'WaistPitch': 100., 'WasitYaw': 100.,
                        'ShoulderPitch': 100., 'ShoulderRoll': 100., 'ShoulderYaw': 100.,
                        'ElbowRoll': 100., 
                        'WristYaw': 100., 'WristRoll': 100.,
                        'HandYaw': 100.,
                        }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/jet/urdf/jet.urdf'
        name = "jet"
        foot_name = 'AnkleRoll'
        terminate_after_contacts_on = ['Hip', 'Waist', 'Shoulder', 'Elbow', 'Wrist', 'Hand', 'Hip'] # check legged_robot.py->_create_envs->uses links of which names contain "terminate_after_contacts_on"
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 700.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_ang_vel = 1.0
            torques = -5.e-6
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            feet_air_time = 5.
            dof_pos_limits = -1.    
            no_fly = 0.5
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.

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
        max_iterations = 1000 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

class JetRoughCfgPPO( LeggedRobotCfgPPO):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_jet'

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01



