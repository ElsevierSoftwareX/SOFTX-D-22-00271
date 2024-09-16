# -*- coding: utf-8 -*-
"""
**RosenPy: An Open Source Python Framework for Complex-Valued Neural Networks**.
*Copyright © A. A. Cruz, K. S. Mayer, D. S. Arantes*.

*License*

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
This file contains a function for applying L2 regularization.
"""

def l2_regularization(module, lambda_init, reg_strength, epoch):
    """
    Calculates the L2 regularization factor for a given epoch.

    Parameters
    ----------
    module : module
        The backend module (e.g., NumPy or CuPy).
    lambda_init : float
        The initial lambda value.
    reg_strength : float
        The regularization strength.
    epoch : int
        The current epoch of the training process.
     
    Returns
    -------
    float
        The L2 regularization factor based on the given parameters.
    """
    return module.multiply(lambda_init, module.exp(-reg_strength * epoch))
