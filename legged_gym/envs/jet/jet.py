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
    
    def _reward_upper_motion(self):
        
        return 0.
    
    
    
        #######NOTE#######
        # The jacobian maps the joint velocities to the body's CoM frame velocities
        # The body's CoM frame is determined by the URDF files.
        # The spatial link velocities that the matrix maps to are with respect to the center of mass (COM) of the links, 
        # and are stacked [vx; vy; vz; wx; wy; wz], where vx and wx refer to the linear and rotational velocity in world X, respectively.
        # The body's origin frame is the joint's frame which connects the link to its parent.
        # The body's CoM frame is described in the URDF file under the tag <inertial>
        # The <origin> tag describes the CoM frame's linear and angular offset from the body's origin frame
        # Therefore, all we need to do is follow the centroidal dynamics paper, since all values are coordinated with the paper.
        # The mass matrix's element for the virtual joints 
        # Note that rpy = 0 0 0 in all links in jet urdf.
        # rb_props.inertia has three methods; x, y, and z.
    
    def _update_inertia(self):
        self.inertia.zero_()
        for key, N in self.body_names_dict.items():
            mass = self.rb_mass[:, N]
            inertia = self.rb_props[:, N, 1:, :]

            self.inertia[:, 6*N+3:6*N+6, 6*N+3:6*N+6] = inertia
            self.inertia[:, 6*N, 6*N] = mass
            self.inertia[:, 6*N+1, 6*N+1] = mass
            self.inertia[:, 6*N+2, 6*N+2] = mass
        # Note that the inertia matrix is much more simple than in the paper
        # This is because the jacobian from PhysX maps the joint velocities
        # to spatial velocities IN COM FRAME.
        
            
    def _update_centroidal_dynamics(self):
        '''
        Now we have the system momentum matrix that maps the joint velocities
        to the angular momentums of each link IN THEIR COM FRAMES
        Note that in the paper, the SMM maps joint velocities to the angular
        momentums of each link IN THEIR JOINT FRAME
        So, we must compute the projection matrices that project
        twists from link-COM frame to centroidal frame.
        To do so, we need 
          1. the rotation matrix from the inertial frame to the COM frame
              Note that the rotation from joint frame to COM frame are all identity
              according to the jet URDF.
              Therefore this is not a problem
          2. the vector from the robot-COM to the link-COM
            we can use rb_states to compute the robot-COM and the link-COM in inertial frame
            The problem is, we do not know if rb_states gives us the
            position of the link-COM, or the link-origin(joint frame)
            We can find out using the world link and the base link.
            The base link has offset from its link frame origin.
            The world link's frame is the same as the base's link frame origin
            print both values.
            If the two are the same, then rb_states gives us the joint frame position
            If not, then the COM frame.
            Turns out, it gives us the joint frame!!
            
            Now, are we sure if the jacobian maps to the COM frame?
            Check the jacobian itself, and if the nth joint for the nth link mapping
            is all 0, which means the EE is at the origin of joint frame,
            then the jacobian maps to joint frame!!
            Checked the last 6 rows of the jacobian, and all the joints affect 
            the spatial velocity -> COM frame
        '''
        self.com_position.zero_() # reset com position
        
        transform = torch.zeros((self.num_envs, 6*self.num_bodies, 6), device=self.device)
        unflatten = gymtorch.torch.nn.Unflatten(0, (self.num_envs, self.num_bodies))
        rb_states = unflatten(self.rb_states) # states are in inertial frame, about the joint frame
        
        tmp_r = torch.zeros((self.num_envs, len(self.body_names_dict), 3, 3), device=self.device) # link rotation matrix in inertial frame
        tmp_com = torch.zeros((self.num_envs, len(self.body_names_dict), 3), device=self.device) # link com position in inertial frame
        tmp_mass = torch.zeros((self.num_envs, 3), device=self.device) # shape is (num_envs, 3) to use in CoM computation
        
        # get CoM position
        for key, N in self.body_names_dict.items():
            quats = rb_states[:, N, 3:7] # G_R_i
            tmp_r[:, N, 0, 0] = 2*(quats[:, 0] * quats[:, 0] + quats[:, 1] * quats[:, 1])-1
            tmp_r[:, N, 0, 1] = 2*(quats[:, 1] * quats[:, 2] - quats[:, 0] * quats[:, 3])
            tmp_r[:, N, 0, 2] = 2*(quats[:, 1] * quats[:, 3] + quats[:, 0] * quats[:, 2])
            tmp_r[:, N, 1, 0] = 2*(quats[:, 1] * quats[:, 2] + quats[:, 0] * quats[:, 3])
            tmp_r[:, N, 1, 1] = 2*(quats[:, 0] * quats[:, 0] + quats[:, 2] * quats[:, 2])-1
            tmp_r[:, N, 1, 2] = 2*(quats[:, 2] * quats[:, 3] - quats[:, 0] * quats[:, 1])
            tmp_r[:, N, 2, 0] = 2*(quats[:, 1] * quats[:, 3] - quats[:, 0] * quats[:, 2])
            tmp_r[:, N, 2, 1] = 2*(quats[:, 2] * quats[:, 3] + quats[:, 0] * quats[:, 1])
            tmp_r[:, N, 2, 2] = 2*(quats[:, 0] * quats[:, 0] + quats[:, 3] * quats[:, 3])-1
            tmp_com[:, N, :] = rb_states[:, N, 0:3] + (tmp_r[:, N, :, :] @ self.rb_props[:, N, 0, :].reshape(self.num_envs, 3, 1)).flatten(1,2)
            self.com_position += self.rb_mass[:, N].reshape(2,1).repeat(1,3) * tmp_com[:, N, :]
            tmp_mass += self.rb_mass[:, N].reshape(2,1).repeat(1,3)
        self.com_position /= tmp_mass
        
        # get Transform
        Sc = gymtorch.torch.zeros((self.num_envs, 3, 3), device=self.device) # skew of c
        for key, N in self.body_names_dict.items():
            # c is the vector from robot-COM to link-COM
            c = tmp_com[:, N, :] - self.com_position
            Sc[:, 0, 1] = - c[:, 2]
            Sc[:, 0, 2] = c[:, 1]
            Sc[:, 1, 2] = -c[:, 0]
            Sc[:, 1, 0] = c[:, 2]
            Sc[:, 2, 0] = -c[:, 1]
            Sc[:, 2, 1] = c[:, 0]          
            tmp = gymtorch.torch.transpose(tmp_r[:, N, :, :], 1, 2) # i_R_G     
            transform[:, 6*N:6*N+3, 0:3] = tmp
            transform[:, 6*N:6*N+3, 3:6] = tmp @ Sc.transpose(1,2)
            transform[:, 6*N+3:6*N+6, 3:6] = tmp             
            
        CMM = transform.transpose(1,2) @ self.inertia @ self.jacobian
        dofvel = gymtorch.torch.cat((rb_states[:, 0, 7:13], self.dof_vel), dim=1)
        self.centroidal_momentum = CMM @ dofvel.reshape(self.num_envs, -1, 1)
      
    def _init_buffers(self):
        super()._init_buffers()
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)        
        self.jacobian = gymtorch.wrap_tensor(_jacobian).flatten(1,2) # originally shape of (num_envs, num_bodies, 6, num_dofs+6)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        
        self.inertia = gymtorch.torch.zeros((self.num_envs, 6*self.num_bodies, 6*self.num_bodies), device = self.device)
        self.centroidal_momentum = gymtorch.torch.zeros((self.num_envs, 6), device=self.device)        
        self.rb_props = gymtorch.torch.zeros((self.num_envs, self.num_bodies, 4, 3), device=self.device) # [comX, comY, comZ], [Ix, Iy, Iz]
        self.rb_mass = gymtorch.torch.zeros((self.num_envs, self.num_bodies), device=self.device) # link mass
        self.com_position = gymtorch.torch.zeros((self.num_envs, 3), device=self.device)
        
        # Reconstruct rb_props as tensor        
        for env in range(self.num_envs):
            for _, N in self.body_names_dict.items():
                rb_props = self.gym.get_actor_rigid_body_properties(self.envs[env], 0)[N]
                # inertia tensors are about link's CoM frame
                self.rb_props[env, N, 0, :] = gymtorch.torch.tensor([rb_props.com.x, rb_props.com.y, rb_props.com.z], device=self.device)
                self.rb_props[env, N, 1, :] = gymtorch.torch.tensor([rb_props.inertia.x.x, rb_props.inertia.x.y, rb_props.inertia.x.z], device=self.device)
                self.rb_props[env, N, 2, :] = gymtorch.torch.tensor([rb_props.inertia.y.x, rb_props.inertia.y.y, rb_props.inertia.y.z], device=self.device)
                self.rb_props[env, N, 3, :] = gymtorch.torch.tensor([rb_props.inertia.z.x, rb_props.inertia.z.y, rb_props.inertia.z.z], device=self.device)

                self.rb_mass[env, N] = rb_props.mass
        
        # Update dynamics        
        self._update_inertia()
        self._update_centroidal_dynamics()
        
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._update_inertia()
        self._update_centroidal_dynamics()
        