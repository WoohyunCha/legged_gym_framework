from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class Jet(LeggedRobot):
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    def _reward_angular_momentum(self):
        
    
        return 0.
    
    
    def _update_inertia(self):
        inv_massmatrix = gymtorch.torch.linalg.inv(self.massmatrix + 1.e-9 * gymtorch.torch.eye(self.num_dofs + 6, device=self.device))    
        for key, value in self.body_names_dict.items():
            tmp = self.jacobian[:, 6*value:6*(value+1), :]
            tmp_t = gymtorch.torch.transpose(tmp, 1, 2) 
            self.inertia[:, 6*value:6*(value+1), 6*value:6*(value+1)] = gymtorch.torch.linalg.inv((tmp @ inv_massmatrix) @ tmp_t)
            # world, base, ... links
            # world link has 0 mass

    def _update_centroidal_dynamics(self):
        transform = torch.zeros((self.num_envs, 6*self.num_bodies, 6), device=self.device)
        unflatten = gymtorch.torch.nn.Unflatten(0, (self.num_envs, self.num_bodies))
        rb_tmp = unflatten(self.rb_states)
        tmp_r = torch.zeros((self.num_envs, 3, 3), device=self.device)
        tmp_com = torch.zeros(self.num_envs, 3, device=self.device)
        mass = 0
        # get CoM position
        for key, value in self.body_names_dict.items():        
            poss = rb_tmp[:, value, 0:3].flatten(1)
            tmp_com += poss * self.gym.get_actor_rigid_body_properties(self.envs[0], 0)[value].mass
            mass += self.gym.get_actor_rigid_body_properties(self.envs[0], 0)[value].mass
        tmp_com = tmp_com / mass
        #get rotations
        Sc = torch.zeros(self.num_envs, 3, 3, device=self.device) # skew symmetric of c vector
        for key, value in self.body_names_dict.items():
            quats = rb_tmp[:, value, 3:7].flatten(1)
            tmp_r[:, 0, 0] = 2*(quats[:, 0] * quats[:, 0] + quats[:, 1] * quats[:, 1])-1
            tmp_r[:, 0, 1] = 2*(quats[:, 1] * quats[:, 2] - quats[:, 0] * quats[:, 3])
            tmp_r[:, 0, 2] = 2*(quats[:, 1] * quats[:, 3] + quats[:, 0] * quats[:, 2])
            tmp_r[:, 1, 0] = 2*(quats[:, 1] * quats[:, 2] + quats[:, 0] * quats[:, 3])
            tmp_r[:, 1, 1] = 2*(quats[:, 0] * quats[:, 0] + quats[:, 2] * quats[:, 2])-1
            tmp_r[:, 1, 2] = 2*(quats[:, 2] * quats[:, 3] - quats[:, 0] * quats[:, 1])
            tmp_r[:, 2, 0] = 2*(quats[:, 1] * quats[:, 3] - quats[:, 0] * quats[:, 2])
            tmp_r[:, 2, 1] = 2*(quats[:, 2] * quats[:, 3] + quats[:, 0] * quats[:, 1])
            tmp_r[:, 2, 2] = 2*(quats[:, 0] * quats[:, 0] + quats[:, 3] * quats[:, 3])-1
            tmp_r = gymtorch.torch.transpose(tmp_r, 1, 2) # i_R_G
            poss = rb_tmp[:, value, 0:3].flatten(1)            
            c = poss - tmp_com
            Sc[:, 0, 1] = - c[:, 2]
            Sc[:, 0, 2] = c[:, 1]
            Sc[:, 1, 2] = -c[:, 0]
            Sc[:, 1, 0] = c[:, 2]
            Sc[:, 2, 0] = -c[:, 1]
            Sc[:, 2, 1] = c[:, 0]           
            Sc = gymtorch.torch.transpose(Sc, 1, 2) 
            transform[:, 6*value:6*value+3, 0:3] = tmp_r @ Sc
            transform[:, 6*value:6*value+3, 3:6] = tmp_r
            transform[:, 6*value+3:6*value+6, 0:3] = tmp_r
            # get CoM position
            # get vector from CoM to link frame, in inertial coordinates
            # use tmp_r and this to compute transform for each link, G_X_i
            #self.transform[:, 6*value:6*(value+1), :] = 1
        CMM = gymtorch.torch.transpose(transform, 1, 2) @ self.inertia @ self.jacobian
        dofvel = gymtorch.torch.zeros((self.num_envs, self.num_dofs + 6), device=self.device)
        dofvel[:, 6:] = self.dof_vel
        dofvel[:, 0:6] = rb_tmp[:, 0, 7:13]
        unflatten = gymtorch.torch.nn.Unflatten(1, (-1, 1))
        dofvel = unflatten(dofvel)
        self.centroidal_momentum = CMM @ dofvel
      
    def _init_buffers(self):
        super()._init_buffers()
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, self.cfg.asset.name)
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.massmatrix = gymtorch.torch.zeros((self.num_envs, self.num_dofs + 6, self.num_dofs + 6) ,device=self.device)
        self.jacobian = gymtorch.wrap_tensor(_jacobian).flatten(1,2)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        
        self.massmatrix[:, 6:, 6:] = gymtorch.wrap_tensor(_massmatrix)        
        self.massmatrix[:, 0:3, 0:3] = 3.90994 * gymtorch.torch.eye(3)
        self.massmatrix[:, 3, 3] = 0.02724195938
        self.massmatrix[:, 4, 4] = 0.00634115870
        self.massmatrix[:, 5, 5] = 0.02399190431
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.inertia = torch.zeros((self.num_envs, 6*self.num_bodies, 6*self.num_bodies), device = self.device)
        self.centroidal_momentum = torch.zeros((self.num_envs, 6), device=self.device)
        self._update_inertia()
        self._update_centroidal_dynamics()
        
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._update_inertia()
        self._update_centroidal_dynamics()
        