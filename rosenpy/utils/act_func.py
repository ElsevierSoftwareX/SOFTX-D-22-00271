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

import numpy as np

def sinh(module, x, derivative=False):
    """
    Activation function - Hyperbolic sine, element-wise.

    Parameters
    ----------
    module : module
        Backend module, typically np or cp (CuPy).
    x : array_like
        Input array.
    derivative : bool, optional
        Whether to compute the derivative. Default is False.
   
    Returns
    -------
    array_like
        Either the activation function (feedforward) or its derivative (backpropagation).
    """
    if derivative:
        return module.cosh(x)
    return module.sinh(x)

def atanh(module, x, derivative=False):
    """
    Activation function - Inverse hyperbolic tangent, element-wise.
    
    Parameters
    ----------
    module : module
        Backend module, typically np or cp (CuPy).
    x : array_like
        Input array.
    derivative : bool, optional
        Whether to compute the derivative. Default is False.
    
    Returns
    -------
    array_like
        Either the activation function (feedforward) or its derivative (backpropagation).
    """
    if derivative:
        return 1 / (1 - module.square(x))
    return module.arctanh(x)

def asinh(module, x, derivative=False):
    """
    Activation function - Inverse hyperbolic sine, element-wise.

    Parameters
    ----------
    module : module
        Backend module, typically np or cp (CuPy).
    x : array_like
        Input array.
    derivative : bool, optional
        Whether to compute the derivative. Default is False.
    
    Returns
    -------
    array_like
        Either the activation function (feedforward) or its derivative (backpropagation).
    """
    if derivative:
        return 1 / (1 + module.square(x))
    return module.arcsinh(x)

def tan(module, x, derivative=False):
    """
    Activation function - Tangent, element-wise.

    Parameters
    ----------
    module : module
        Backend module, typically np or cp (CuPy).
    x : array_like
        Input array.
    derivative : bool, optional
        Whether to compute the derivative. Default is False.
    
    Returns
    -------
    array_like
        Either the activation function (feedforward) or its derivative (backpropagation).
    """
    if derivative:
        return 1 / module.square(module.cos(x))
    return module.tan(x)

def sin(module, x, derivative=False):
    """
    Activation function - Sine, element-wise.

    Parameters
    ----------
    module : module
        Backend module, typically np or cp (CuPy).
    x : array_like
        Input array.
    derivative : bool, optional
        Whether to compute the derivative. Default is False.
    
    Returns
    -------
    array_like
        Either the activation function (feedforward) or its derivative (backpropagation).
    """
    if derivative:
        return module.cos(x)
    return module.sin(x)

def atan(module, x, derivative=False):
    """
    Activation function - Arc tangent, element-wise.

    Parameters
    ----------
    module : module
        Backend module, typically np or cp (CuPy).
    x : array_like
        Input array.
    derivative : bool, optional
        Whether to compute the derivative. Default is False.
    
    Returns
    -------
    array_like
        Either the activation function (feedforward) or its derivative (backpropagation).
    """
    if derivative:
        return 1 / (1 + module.square(x))
    return module.arctan(x)

def asin(module, x, derivative=False):
    """
    Activation function - Arc sine, element-wise.

    Parameters
    ----------
    module : module
        Backend module, typically np or cp (CuPy).
    x : array_like
        Input array.
    derivative : bool, optional
        Whether to compute the derivative. Default is False.
    
    Returns
    -------
    array_like
        Either the activation function (feedforward) or its derivative (backpropagation).
    """
    if derivative:
        return 1 / module.sqrt(1 - module.square(x))
    return module.arcsin(x)

def acos(module, x, derivative=False):
    """
    Activation function - Arc cosine, element-wise.

    Parameters
    ----------
    module : module
        Backend module, typically np or cp (CuPy).
    x : array_like
        Input array.
    derivative : bool, optional
        Whether to compute the derivative. Default is False.
    
    Returns
    -------
    array_like
        Either the activation function (feedforward) or its derivative (backpropagation).
    """
    if derivative:
        return 1 / module.sqrt(module.square(x) - 1)
    return module.arccos(x)

def sech(module, x, derivative=False):
    """
    Activation function - Hyperbolic secant, element-wise.

    Parameters
    ----------
    module : module
        Backend module, typically np or cp (CuPy).
    x : array_like
        Input array.
    derivative : bool, optional
        Whether to compute the derivative. Default is False.
    
    Returns
    -------
    array_like
        Either the activation function (feedforward) or its derivative (backpropagation).
    """
    ex = module.exp(x)
    if derivative:
        return -2 * ex / (ex + 1 / ex) ** 2
    return 2 / (ex + 1 / ex)

def linear(module, x, derivative=False):
    """
    Linear activation function, also known as "no activation" or "identity function."

    Parameters
    ----------
    module : module
        Backend module, typically np or cp (CuPy).
    x : array_like
        Input array.
    derivative : bool, optional
        Whether to compute the derivative. Default is False.
    
    Returns
    -------
    array_like
        Either the activation function (feedforward) or its derivative (backpropagation).
    """
    return module.ones_like(x) if derivative else x

def tanh(module, x, derivative=False):
    """
    Activation function - Hyperbolic tangent, element-wise.

    Parameters
    ----------
    module : module
        Backend module, typically np or cp (CuPy).
    x : array_like
        Input array.
    derivative : bool, optional
        Whether to compute the derivative. Default is False.
    
    Returns
    -------
    array_like
        Either the activation function (feedforward) or its derivative (backpropagation).
    """
    if derivative:
        return 1 - module.square(module.tanh(x))
    return module.tanh(x)

def split_complex(module, y, act_func, derivative=False):
    """
    Applies activation functions separately to the real and imaginary components of a complex input.

    Parameters
    ----------
    module : module
        Backend module, typically np or cp (CuPy).
    y : array_like
        Input array.
    act_func : function
        Activation function to be applied to both components.
    derivative : bool, optional
        Whether to compute the derivative. Default is False.
    
    Returns
    -------
    array_like
        Result of applying the activation function (or its derivative) to the real and imaginary parts.
    """
    return act_func(module, module.real(y), derivative) + 1.0j * act_func(module, module.imag(y), derivative)
