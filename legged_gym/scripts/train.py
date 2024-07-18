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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils.helpers import get_load_path
import torch

from collections import deque
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, custom_task_registry, Logger
import matplotlib.pyplot as plt


def deque_to_tensor(buffer : deque) -> torch.Tensor:
    if not buffer:
        raise ValueError("Deque is empty. No data to change into tensor.")
    if torch.is_tensor(buffer[0]) is not True:
        raise TypeError("Given deque does not contain torch tensors.")
    for i, obs in enumerate(buffer):
        if i == 0:
            ret = obs
            ret.to(obs.device)
        else:
            ret = torch.cat((ret, obs), dim=1)
    if ret.shape[0] != buffer[0].shape[0] or ret.shape[1] != buffer[0].shape[1]*len(buffer):
        raise ValueError("Conversion from deque to tensor is wrong.")
    return ret

def train(args):
    if args.task in task_registry.task_classes:
        env, env_cfg = task_registry.make_env(name=args.task, args=args) # env is registed + args
        registry = task_registry
    elif args.task in custom_task_registry.task_classes:
        env, env_cfg = custom_task_registry.make_env(name=args.task, args=args) # env is registed + args
        registry = custom_task_registry
    
    env_cfg_play, train_cfg_play = registry.get_cfgs(name=args.task)
    history_len = train_cfg_play.runner.history_len
    ppo_runner, train_cfg = registry.make_alg_runner(env=env, name=args.task, args=args) # if task_cfg = [some dict file], alg_runner ignores nameCfgPPO and uses task_cfg file    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':

    # Play once for logging
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    SAVE_FIG = True
    args = get_args()
    train(args)
