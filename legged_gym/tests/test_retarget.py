import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils.reference_mapper import reference_mapper
from legged_gym.utils.helpers import txt_to_numpy

reference_mapper(None, None)
reference_folder = os.path.abspath('/home/cha/isaac_ws/legged_gym/resources/reference_motions/')
name_of_reference = "processed_data_tocabi_walk.txt"
reference = txt_to_numpy(os.path.join(reference_folder, name_of_reference))
print(reference.shape)

###AMP refernece motion check###
# name_of_reference = "amp_humanoid_walk"

# path_to_reference = os.path.join(reference_folder, name_of_reference+'.npy')

# reference = np.load(path_to_reference, allow_pickle=True).item()
# for key, val in reference.items():
#     print(key)
#     print(type(val))
#     print('--------------')
# print('rotation shape : ', reference['rotation']['arr'].shape)
# print("root translation shape : ", reference['root_translation']['arr'].shape)
# print("global velocity shape : ", reference['global_velocity']['arr'].shape)
# print("global angular velocity : ", reference['global_angular_velocity']['arr'].shape)
# print("fps : ", reference['fps'])