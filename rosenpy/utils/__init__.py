# -*- coding: utf-8 -*-
"""RosenPy: An Open Source Python Framework for Complex-Valued Neural Networks.
Copyright © A. A. Cruz, K. S. Mayer, D. S. Arantes.

License:

This file is part of RosenPy.
RosenPy is an open source framework distributed under the terms of the GNU General 
Public License, as published by the Free Software Foundation, either version 3 of 
the License, or (at your option) any later version. For additional information on 
license terms, please open the Readme.md file.

RosenPy is distributed in the hope that it will be useful to every user, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details. 

You should have received a copy of the GNU General Public License
along with RosenPy. If not, see <http://www.gnu.org/licenses/>.
"""

"""
This file contains all the activation functions used by RosenPy.
"""

# utils/__init__.py

# Importing activation functions
from .act_func import sinh, atanh, asinh, tan, sin, atan, asin, acos, sech, linear, tanh, split_complex

# Importing batch generation functions
from .batch_gen_func import batch_sequential, batch_shuffle

# Importing cost functions
from .cost_func import mse

# Importing learning rate decay functions
from .decay_func import none_decay, time_based_decay, exponential_decay, staircase

# Importing weight initialization functions
from .init_func import zeros, zeros_real, ones, ones_real, random_normal, random_uniform, glorot_normal, glorot_uniform, rbf_default, ru_gamma_ptrbf, ru_weights_ptrbf, opt_ptrbf_gamma, opt_ptrbf_weights, opt_conv_ptrbf_weights, opt_crbf_gamma, opt_crbf_weights

# Importing regularization functions
from .reg_func import l2_regularization

# Importing module selection function
from .select_module import select_module

# Importing utility functions
from .utils import split_set
