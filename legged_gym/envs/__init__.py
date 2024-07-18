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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR # project root(isaacgym/legged_gym, because setup.py is here) -> legged_gym. By from legged_gym, the __init__.py is executed. 
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot # relative directory used
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .jet.jet import Jet
from .jet.jet_rough_config import JetRoughCfg, JetRoughCfgPPO
from .jet.jet_flat_config import JetFlatCfg, JetFlatCfgPPO
from .bolt6.bolt6 import Bolt6
from .bolt6.bolt6_config import Bolt6Cfg, Bolt6CfgPPO
from .bolt10.bolt10 import Bolt10
from .bolt10.bolt10_config import Bolt10Cfg, Bolt10CfgPPO
from .tocabi.tocabi import Tocabi
from .tocabi.tocabi_config import TocabiCfg, TocabiCfgPPO
from .g1.g1 import g1
from .g1.g1_config import g1Cfg, g1CfgPPO



import os

from legged_gym.utils.task_registry import task_registry # task_registry is a global registry, initialized in legged_gym.utils.task_registry.py
from legged_gym.utils.custom_task_registry import custom_task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() ) # AnymalCRoughCfg() is an instance of LeggedRobotCfg
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() ) # Anymal is a VecEnv object (env)
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )

task_registry.register("jet_rough", Jet, JetRoughCfg(), JetRoughCfgPPO())
task_registry.register("jet_flat", Jet, JetFlatCfg(), JetFlatCfgPPO())

custom_task_registry.register("bolt10", Bolt10, Bolt10Cfg(), Bolt10CfgPPO())
custom_task_registry.register("bolt6", Bolt6, Bolt6Cfg(), Bolt6CfgPPO())

custom_task_registry.register("tocabi", Tocabi, TocabiCfg(), TocabiCfgPPO())
custom_task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
custom_task_registry.register("g1", g1, g1Cfg(), g1CfgPPO())
