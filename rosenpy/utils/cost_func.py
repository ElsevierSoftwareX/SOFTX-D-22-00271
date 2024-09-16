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

def mse(module, y, y_pred):
    """
    Calculate the mean squared error (MSE) between the target values (y) and the predicted values (y_pred).

    Parameters
    ----------
    module : module
        The CuPy or NumPy module for array operations.
    y : array-like
        The target values.
    y_pred : array-like
        The predicted values.

    Returns
    -------
    float
        The mean squared error between y and y_pred.
    """
    error = module.square(module.abs(y - y_pred))
    return (0.5 * module.dot(module.ones((1, error.shape[0])), error) / error.shape[0])[0][0]
