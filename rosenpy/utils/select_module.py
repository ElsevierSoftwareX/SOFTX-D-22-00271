# -*- coding: utf-8 -*-
"""
RosenPy: An Open Source Python Framework for Complex-Valued Neural Networks.
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

import numpy as np


def select_module(use_gpu):
    """
    This function is used to implement CPU/GPU generic code.

    Parameters
    ----------
    use_gpu : bool
        A flag indicating whether to use GPU acceleration (CuPy) or CPU (NumPy).

    Returns
    -------
    module : module
        Returns the module `cupy` if available and requested, otherwise `numpy`.

    """
    if use_gpu:
        try:
            import cupy as module
            print("CuPy module selected for GPU computation.")
        except ImportError:
            print("CuPy is not installed. Falling back to NumPy for CPU computation.")
            module = np
    else:
        print("NumPy module selected for CPU computation.")
        module = np

    return module
