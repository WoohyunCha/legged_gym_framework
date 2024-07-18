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

from time import time
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.custom_terrain import custom_Terrain

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

from collections import deque
from legged_gym.utils.helpers import txt_to_numpy
from legged_gym.utils.torch_jit_utils import cubic, quat_diff_rad

from legged_gym import LEGGED_GYM_ROOT_DIR


class Tocabi(LeggedRobot):
 

    def _reward_mimic_body_orientation(self):
        torso_rot = self.root_states[:, 3:7]
        Identity_rot = torch.zeros_like(torso_rot)
        Identity_rot[..., -1] = 1.
        return torch.exp(-13.2 * torch.abs(quat_diff_rad(Identity_rot, torso_rot)))
    
    def _reward_qpos_regulation(self):
        return torch.exp(-2.0 * torch.norm((self.target_data_qpos[:,0:self.num_actuations] - self.dof_pos[:,0:]), dim=1)**2)
    
    def _reward_qvel_regulation(self):
        return torch.exp(-0.01 * torch.norm((torch.zeros_like(self.dof_vel[:, 0:self.num_actuations]) - self.dof_vel[:,0:]), dim=1)**2)

    def _reward_contact_force_penalty(self):
        lfoot_force = self.feet_force[:, 0, :]
        rfoot_force = self.feet_force[:, 1, :]
        left_foot_thres = lfoot_force[:,2].unsqueeze(-1) > 1.4*9.81*self.robot_mass
        right_foot_thres = rfoot_force[:,2].unsqueeze(-1) > 1.4*9.81*self.robot_mass
        thres = left_foot_thres | right_foot_thres
        contact_force_penalty_thres = 0.1*(1-torch.exp(-0.007*(torch.norm(torch.clamp(lfoot_force[:,2].unsqueeze(-1) - 1.4*9.81*self.robot_mass, min=0.0), dim=1) \
                                                            + torch.norm(torch.clamp(rfoot_force[:,2].unsqueeze(-1) - 1.4*9.81*self.robot_mass, min=0.0), dim=1))))
        return torch.where(thres.squeeze(-1), contact_force_penalty_thres[:], 0.1*torch.ones_like(contact_force_penalty_thres[:])[:])
    
    def _reward_torque_regulation(self):
        return torch.exp(-0.01 * torch.norm(self.torques,dim=1)) # TODO
    
    def _reward_torque_diff_regulation(self):
        return torch.exp(-0.01 * torch.norm((self.torques - self.last_torques), dim=1))
    
    def _reward_qacc_regulation(self):
        return torch.exp(-20.0*torch.norm((self.dof_vel[:,0:]-self.last_dof_vel[:,0:]), dim=1)**2)
    
    def _reward_body_vel(self):
        return torch.exp(-3.0 * torch.norm((self.commands[:,0:2] - self.root_states[:,7:9]), dim=1)**2)
    
    def _reward_foot_contact(self):
        DSP = (3300 <= self.reference_data_idx) & (self.reference_data_idx < 3600) 
        DSP = DSP | (self.reference_data_idx < 300) 
        DSP = DSP | ((1500 <= self.reference_data_idx) & ( self.reference_data_idx < 2100))
        RSSP = (300 <= self.reference_data_idx) & (self.reference_data_idx < 1500)
        LSSP = (2100 <= self.reference_data_idx) & (self.reference_data_idx < 3300)
        lfoot_force = self.feet_force[:, 0, :]
        rfoot_force = self.feet_force[:, 1, :]
        left_foot_contact = (lfoot_force[:,2].unsqueeze(-1) > 1.)
        right_foot_contact = (rfoot_force[:,2].unsqueeze(-1) > 1.)
        DSP_sync = DSP & right_foot_contact & left_foot_contact
        RSSP_sync = RSSP & right_foot_contact & ~left_foot_contact
        LSSP_sync = LSSP & ~right_foot_contact & left_foot_contact
        foot_contact_reward = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False)
        foot_contact_feeder = 0.2*torch.ones_like(foot_contact_reward, dtype=torch.float)
        foot_contact_reward = torch.where(DSP_sync.squeeze(-1), foot_contact_feeder, foot_contact_reward)
        foot_contact_reward = torch.where(RSSP_sync.squeeze(-1), foot_contact_feeder, foot_contact_reward)
        foot_contact_reward = torch.where(LSSP_sync.squeeze(-1), foot_contact_feeder, foot_contact_reward)
        return foot_contact_reward
    


    def _reward_contact_force_diff(self):
        lfoot_force = self.feet_force[:, 0, :]
        rfoot_force = self.feet_force[:, 1, :]
        lfoot_force_pre = self.last_feet_force[:, 0, :]
        rfoot_force_pre = self.last_feet_force[:, 1, :]
        return torch.exp(-0.01*(torch.norm(lfoot_force[:]-lfoot_force_pre[:], dim=1) + \
                                                            torch.norm(rfoot_force[:]-rfoot_force_pre[:], dim=1)))        


    def _reward_force_diff_thres_penalty(self):
        lfoot_force = self.feet_force[:, 0, :]
        rfoot_force = self.feet_force[:, 1, :]        
        lfoot_force_pre = self.last_feet_force[:, 0, :]
        rfoot_force_pre = self.last_feet_force[:, 1, :]
        left_foot_thres_diff = torch.abs(lfoot_force[:,2]-lfoot_force_pre[:,2]).unsqueeze(-1) > 0.2*9.81*self.robot_mass
        right_foot_thres_diff = torch.abs(rfoot_force[:,2]-rfoot_force_pre[:,2]).unsqueeze(-1) > 0.2*9.81*self.robot_mass
        thres_diff = left_foot_thres_diff | right_foot_thres_diff
        force_diff_thres_penalty = torch.where(thres_diff.squeeze(-1), -0.05*torch.ones(self.num_envs, device=self.device, requires_grad=False)[:], torch.zeros(self.num_envs, device=self.device, requires_grad=False)[:])      
        return force_diff_thres_penalty

    def _reward_force_thres_penalty(self):
        lfoot_force = self.feet_force[:, 0, :]
        rfoot_force = self.feet_force[:, 1, :]
        left_foot_thres = lfoot_force[:,2].unsqueeze(-1) > 1.4*9.81*self.robot_mass
        right_foot_thres = rfoot_force[:,2].unsqueeze(-1) > 1.4*9.81*self.robot_mass
        thres = left_foot_thres | right_foot_thres
        return torch.where(thres.squeeze(-1), -0.2*torch.ones(self.num_envs, device=self.device, requires_grad=False)[:], torch.zeros(self.num_envs, device=self.device, requires_grad=False)[:])

    def _reward_force_ref(self):
        weight_scale = self.robot_mass / 104.48
        lfoot_force = self.feet_force[:, 0, :]
        rfoot_force = self.feet_force[:, 1, :]        
        return torch.exp(-0.001*(torch.abs(lfoot_force[:,2]+weight_scale.squeeze(-1)*self.target_data_force[:,0]))) +\
                        torch.exp(-0.001*(torch.abs(rfoot_force[:,2]+weight_scale.squeeze(-1)*self.target_data_force[:,1])))

    def _reward_alive(self):
        return self.commands[:, 0]

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.exp(-5e-3 * torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=-1))    
    
    def _reward_energy_minimization(self):
        std = 1e-4
        # return torch.exp(std*torch.clamp(-torch.sum(self.torques * self.dof_vel, dim=-1), max=0))
        return -torch.sum(self.torques * self.dof_vel, dim=-1)
    
    def _reward_imitate_q(self):
        return torch.exp(-2.0 * torch.norm((self.target_data_qpos[:,0:self.num_actuations] - self.dof_pos[:,0:self.num_actuations]), dim=1)**2)

    def _custom_init(self, cfg: LeggedRobotCfg):
        self.num_actuations = cfg.env.num_actuations
        self.control_tick = torch.zeros(
            self.num_envs, 1, dtype=torch.int,
            device=self.device, requires_grad=False)
        self.ext_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.ext_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        if self.num_privileged_obs is not None:
            self.dof_props = torch.zeros((self.num_dofs, 2), device=self.device, dtype=torch.float) # includes dof friction (0) and damping (1) for each environment
        
        self.termination_height = cfg.asset.termination_height
        
        if cfg.asset.reference_data is not None:
            asset_path = self.cfg.asset.reference_data.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            reference_data_non_torch = np.genfromtxt(asset_path,encoding='ascii') 
            self.reference_data = torch.tensor(reference_data_non_torch,device=self.device, dtype=torch.float)
            print("Reference Motion is Loaded. Shape of reference : ", self.reference_data.shape)
            #for Deep Mimic
            self.init_reference_data_idx = torch.zeros(self.num_envs,1,device=self.device, dtype=torch.long)
            self.reference_data_idx = torch.zeros(self.num_envs,1,device=self.device, dtype=torch.long)
            self.reference_data_num = int(self.reference_data.shape[0] - 1)
            self.reference_cycle_dt = self.reference_data[1,0] - self.reference_data[0,0] #0.0005
            self.reference_cycle_period = self.reference_data_num * self.reference_cycle_dt
            self.time = torch.zeros(self.num_envs,1, device=self.device, dtype=torch.float)
        else:
            self.reference_data = None
            print("No Reference Motion is used")
    


    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actuations, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torques = torch.zeros_like(self.torques)
        self.feet_force = torch.zeros((self.num_envs, 2, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.last_feet_force = torch.zeros_like(self.feet_force)
        self.p_gains = torch.zeros(self.num_actuations, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actuations, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) 
        self.action_phase_scale = self.cfg.normalization.action_scales.phase
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()
        
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
                    
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)        
        self.jacobian = gymtorch.wrap_tensor(_jacobian).flatten(1,2) # originally shape of (num_envs, num_bodies, 6, num_dofs+6)
        # The jacobian maps joint velocities (num_dofs + 6) to spatial velocities of CoM frame of each link in global frame
        # https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/docs/Articulations.html#jacobian

        self.rb_states = gymtorch.wrap_tensor(_rb_states).view(self.num_envs, self.num_bodies, 13)
        self.rb_inertia = gymtorch.torch.zeros((self.num_envs, self.num_bodies, 3, 3), device=self.device) # [Ix, Iy, Iz]
        self.rb_mass = gymtorch.torch.zeros((self.num_envs, self.num_bodies), device=self.device) # link mass
        self.rb_com = gymtorch.torch.zeros((self.num_envs, self.num_bodies, 3), device = self.device) # [comX, comY, comZ] in link's origin frame 
        self.com_position = gymtorch.torch.zeros((self.num_envs, 3), device=self.device) # robot-com position in global frame
        
        # Reconstruct rb_props as tensor        
        for env in range(self.num_envs):
            for key, N in self.body_names_dict.items():
                rb_props = self.gym.get_actor_rigid_body_properties(self.envs[env], 0)[N]
                # inertia tensors are about link's CoM frame
                self.rb_com[env, N, :] = gymtorch.torch.tensor([rb_props.com.x, rb_props.com.y, rb_props.com.z], device=self.device)
                self.rb_inertia[env, N, 0, :] = gymtorch.torch.tensor([rb_props.inertia.x.x, -rb_props.inertia.x.y, -rb_props.inertia.x.z], device=self.device)
                self.rb_inertia[env, N, 1, :] = gymtorch.torch.tensor([-rb_props.inertia.y.x, rb_props.inertia.y.y, -rb_props.inertia.y.z], device=self.device)
                self.rb_inertia[env, N, 2, :] = gymtorch.torch.tensor([-rb_props.inertia.z.x, -rb_props.inertia.z.y, rb_props.inertia.z.z], device=self.device)
                # see how inertia tensor is made : https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/dd277ec654440f4c2b5b07d6c286c3fd_MIT16_07F09_Lec26.pdf
                self.rb_mass[env, N] = rb_props.mass
        self.robot_mass = torch.sum(self.rb_mass, dim=1).unsqueeze(1)

        # self.rb_{property} can be used for dynamics calculation

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = custom_Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def destroy_sim(self):
        """Destroy simulation, terrain and environments
        """
        self.gym.destroy_sim(self.sim)
        print("Simulation destroyed")

    def compute_observations(self):
        """ Computes observations
        """
        if self.reference_data is not None:
            pi =  3.14159265358979 
            time2idx = (self.time % self.reference_cycle_period) / self.reference_cycle_dt
            phase = (self.init_reference_data_idx + time2idx) % self.reference_data_num / self.reference_data_num
            sin_phase = torch.sin(2*pi*phase) 
            cos_phase = torch.cos(2*pi*phase)

            self.contacts = self.contact_forces[:, self.feet_indices, 2] > 0.001
            self.obs_buf = torch.cat((  
                                        self.projected_gravity, # 3
                                        self.commands[:, :3] * self.commands_scale, # 3
                                        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12
                                        self.dof_vel * self.obs_scales.dof_vel, # 12
                                        self.actions[:, :self.num_actuations], # 12
                                        sin_phase.view(-1,1), # 1
                                        cos_phase.view(-1,1), # 1
                                    ),dim=-1)
        else:
            self.obs_buf = torch.cat((  
                            self.projected_gravity, # 3
                            self.commands[:, :3] * self.commands_scale, # 3
                            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12
                            self.dof_vel * self.obs_scales.dof_vel, # 12
                            self.actions, # 13
                            ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            self.obs_buf = torch.cat((self.obs_buf, torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements), dim=-1)
        # add noise if needed
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat((
                self.obs_buf,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel  * self.obs_scales.ang_vel,
                # self.contacts,
                # self.contact_forces.squeeze(), # When trimesh, this is unreliable
                self.ext_forces[:, 0, :]* self.obs_scales.ext_forces / self.robot_mass,
                self.ext_torques[:, 0, :]* self.obs_scales.ext_torques/ self.robot_mass,
                self.friction_coeffs* self.obs_scales.friction_coeffs,
                self.dof_props[:, 0].repeat(self.num_envs, 1)*self.obs_scales.dof_friction,
                self.dof_props[:, 1].repeat(self.num_envs, 1)*self.obs_scales.dof_damping,
                torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                ), dim=-1)
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.
        noise_vec[9:19] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[19:29] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[29:39] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[39:] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    def pre_physics_step(self):


        if self.cfg.domain_rand.randomize_dof_friction and  (self.common_step_counter % self.cfg.domain_rand.dof_friction_interval == 0):
            props = self.gym.get_actor_dof_properties(self.envs[0], 0)
            for dof in range(self.num_dofs):
               props["friction"][dof] = np.random.uniform(*self.cfg.domain_rand.dof_friction)
               props["damping"][dof] = np.random.uniform(*self.cfg.domain_rand.dof_damping)
            for envs in self.envs:
                self.gym.set_actor_dof_properties(envs, 0, props)
            if self.num_privileged_obs is not None:
                self.dof_props[:, 0] =  torch.tensor(props["friction"], device=self.device)
                self.dof_props[:, 1] = torch.tensor(props["damping"], device=self.device)
        if self.cfg.domain_rand.ext_force_robots and (self.common_step_counter % self.cfg.domain_rand.ext_force_randomize_interval == 0):
            self.cfg.domain_rand.ext_force_duration = np.ceil(np.random.uniform(*self.cfg.domain_rand.ext_force_duration_s) / self.dt) 

        # Calculate dynamics
        for foot in self.feet_indices:             
            # link rotation matrix, link com position, robot com position computed
            self.footstep_current = self.rb_states[:, foot, :3] + quat_rotate(self.rb_states[:, foot, 3:7], self.rb_com[:, foot, :]) # in global frame
            self.footstep_target = 0 

        # Reference Motion
        if self.reference_data is not None:
            local_time = self.time % self.reference_cycle_period
            local_time_plus_init = (local_time + self.init_reference_data_idx*self.reference_cycle_dt) % self.reference_cycle_period
            self.reference_data_idx = (self.init_reference_data_idx + (local_time / self.reference_cycle_dt).type(torch.long)) % self.reference_data_num
            next_idx = self.reference_data_idx + 1 

            reference_data_idx_list = self.reference_data_idx.squeeze(dim=-1)
            next_idx_list = next_idx.squeeze(dim=-1)

            self.target_data_qpos = cubic(local_time_plus_init, self.reference_data[reference_data_idx_list,0].unsqueeze(-1), self.reference_data[next_idx_list,0].unsqueeze(-1), 
                                            self.reference_data[reference_data_idx_list,1:34], self.reference_data[next_idx_list,1:34], 0.0, 0.0)
            self.target_data_force = cubic(local_time_plus_init, self.reference_data[reference_data_idx_list,0].unsqueeze(-1), self.reference_data[next_idx_list,0].unsqueeze(-1), 
                                            self.reference_data[reference_data_idx_list,34:], self.reference_data[next_idx_list,34:], 0.0, 0.0)
            positive_mask = self.actions[:,-1]>0
            self.actions[:,-1] = positive_mask * self.actions[:,-1]             


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device) # at reset, action is zero.
        self.actions[:, -1] *= self.action_phase_scale
        # step physics and render each frame
        self.pre_physics_step()
        self.render(sync_frame_time=True)


        for _ in range(self.cfg.control.decimation): # compute torque 4 times
            
            if self.cfg.domain_rand.ext_force_robots and  (self.common_step_counter % self.cfg.domain_rand.ext_force_interval < self.cfg.domain_rand.ext_force_duration):  
                self.ext_forces[:, 0, :] = torch.tensor([np.random.uniform(*self.cfg.domain_rand.ext_force_vector_6d_range[i]) for i in range(0,3)], device=self.device, requires_grad=False)    #index: root, body, force axis(6)
                self.ext_torques[:, 0, :] = torch.tensor([np.random.uniform(*self.cfg.domain_rand.ext_force_vector_6d_range[i]) for i in range(3,6)], device=self.device, requires_grad=False)                
            else:
                self.ext_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                self.ext_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)            
            
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.ext_forces), gymtorch.unwrap_tensor(self.ext_torques), gymapi.ENV_SPACE)
            self.last_torques = self.torques
            self.torques = self._compute_torques(self.actions).view(self.torques.shape) 
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques)) 
            self.gym.simulate(self.sim)
            
            # ## set custom state DEBUGGING
            # self.dof_pos[:, :] = 5
            # self.dof_vel[:, :] = -3
            # self.root_states[:, 0:3] = 2
            # self.root_states[:, 3:7] = torch.tensor([.5, .5, .5, .5]) # xyzw
            # self.root_states[:, 7:10] = -1
            # self.root_states[:, 10:13] = -0.2
            # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
            # self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            
        self.post_physics_step()
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions[:, :self.num_actuations] * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)
        # If the agent does not terminate

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed   
        """
        self.last_feet_force = self.feet_force

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.feet_force = self.contact_forces[:, self.feet_indices, :]
        if hasattr(self, "custom_post_physics_step"):
            self.custom_post_physics_step()
            
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def custom_post_physics_step(self):
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)        
        self.control_tick += 1
        self.measured_heights = self._get_heights()

        if self.reference_data is not None:
            self.time += self.dt
            self.time += 5*self.dt*self.actions[:,-1].unsqueeze(-1)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.failure_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.height_buf = self.root_states[:, 2] < self.termination_height[0]
        self.height_buf |= self.root_states[:,2] > self.termination_height[1]
        self.reset_buf |= self.height_buf

    def _custom_reset(self, env_ids):
        self.control_tick[env_ids, 0] = 0   
        self.last_torques[env_ids, :] = 0
        self.last_actions[env_ids, :] = 0
        self.last_dof_vel[env_ids, :] = 0
        self.last_contacts[env_ids, :] = 0
        self.last_root_vel[env_ids, :] = 0

        # Use of reference
        if self.reference_data is not None:
            rand = torch.rand(len(env_ids),1,device=self.device, dtype=torch.float)
            mask = rand > 0.5       
            assert self.reference_data.shape[0]%2 == 0, "Reference data length is not symmetric"
            self.init_reference_data_idx[env_ids] = torch.where(mask, 0, int(self.reference_data.shape[0]/2))
            self.time[env_ids] = 0

        
    def _custom_create_envs(self):
        collision_mask = [] # List of shapes for which collision must be detected
        if self.cfg.domain_rand.randomize_friction:
            # prepare friction randomization
            friction_range = self.cfg.domain_rand.friction_range
            self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)
        else:
            self.friction_coeffs = torch.ones((self.num_envs, 1), device=self.device)
        # for i, env in enumerate(self.envs):
        #     rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env, 0)
        #     for j in range(len(rigid_shape_props)):
        #         if j not in collision_mask:
        #             rigid_shape_props[j].filter=1
        #         rigid_shape_props[j].friction=self.friction_coeffs[i, 0]
        #     self.gym.set_actor_rigid_shape_properties(env, 0, rigid_shape_props)
            
        # for name, num in self.body_names_dict.items():
        #     shape_id = self.body_to_shapes[num]
        #     print("body : ", name, ", shape index start : ", shape_id.start, ", shape index count : ", shape_id.count)
        #     for i in range(shape_id.start, shape_id.start + shape_id.count):
        #         print("shape ", i, " filter : ", self.gym.get_actor_rigid_shape_properties(self.envs[0], 0)[i].filter)
        #         print("shape ", i, " contact offset : ", self.gym.get_actor_rigid_shape_properties(self.envs[0], 0)[i].contact_offset)
                # I guess the best I can try is set the shin's bitmasks as 0           


    def _custom_parse_cfg(self, cfg):
        self.cfg.domain_rand.ext_force_interval = np.ceil(self.cfg.domain_rand.ext_force_interval_s / self.dt)
        self.cfg.domain_rand.ext_force_randomize_interval = np.ceil(self.cfg.domain_rand.ext_force_randomize_interval_s / self.dt)
        self.cfg.domain_rand.dof_friction_interval = np.ceil(self.cfg.domain_rand.dof_friction_interval_s / self.dt)      
        
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props
    
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)    
    
    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer,None, cam_pos, cam_target)
        
        # ##################### HELPER FUNCTIONS ################################## #

    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x)/self.cfg.rewards.tracking_sigma)
