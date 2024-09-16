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
This file contains various initialization functions for initializing complex matrices.
"""

def zeros(module, rows, cols, i=0):
    """
    Initializes a complex matrix with all elements set to zero.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) with all elements set to zero.
    """
    return module.zeros((rows, cols), dtype=complex)


def zeros_real(module, rows, cols, i=0):
    """
    Initializes a real matrix with all elements set to zero.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A real matrix of size (rows, cols) with all elements set to zero.
    """
    return module.zeros((rows, cols), dtype=float)


def ones(module, rows, cols, i=0):
    """
    Initializes a complex matrix with all elements set to one.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) with all elements set to one.
    """
    return module.ones((rows, cols), dtype=complex) + 1j


def ones_real(module, rows, cols, i=0):
    """
    Initializes a real matrix with all elements set to one.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A real matrix of size (rows, cols) with all elements set to one.
    """
    return module.ones((rows, cols), dtype=float)


def random_normal(module, rows, cols, i=0):
    """
    Initializes a complex matrix with elements sampled from a normal distribution.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) with elements sampled from a normal distribution.
    """
    real = module.random.randn(rows, cols).astype(module.float32) - 0.5
    imag = module.random.randn(rows, cols).astype(module.float32) - 0.5
    return (real + 1j * imag) / 10


def random_uniform(module, rows, cols, i=0):
    """
    Initializes a complex matrix with elements sampled from a uniform distribution.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) with elements sampled from a uniform distribution.
    """
    real = module.random.rand(rows, cols).astype(module.float32)
    imag = module.random.rand(rows, cols).astype(module.float32)
    return (real + 1j * imag) / 10


def glorot_normal(module, rows, cols, i=0):
    """
    Initializes a complex matrix using the Glorot normal initialization method.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) initialized using the Glorot normal initialization method.
    """
    std_dev = module.sqrt(2.0 / (rows + cols)) / 10
    return (std_dev * module.random.randn(rows, cols) + 1j * std_dev * module.random.randn(rows, cols)) / 10


def glorot_uniform(module, rows, cols, i=0):
    """
    Initializes a complex matrix using the Glorot uniform initialization method.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) initialized using the Glorot uniform initialization method.
    """
    std_dev = module.sqrt(6.0 / (rows + cols)) / 10
    return (2 * std_dev * module.random.randn(rows, cols) - std_dev + 1j * (std_dev * module.random.randn(rows, cols) - std_dev)) / 5


def rbf_default(module, rows, cols, i=0):
    """
    Initializes a complex matrix with elements generated from a random binary distribution.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) with elements generated from a random binary distribution.
    """
    return module.random.randint(2, size=[rows, cols]) * 0.7 + 1j * (module.random.randint(2, size=[rows, cols]) * 2 - 1) * 0.7


def ru_gamma_ptrbf(module, rows, cols, i=0):
    """
    Initializes a complex matrix with elements sampled from a uniform distribution.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) with elements sampled from a uniform distribution.
    """
    real = module.random.rand(rows, cols).astype(module.float32)
    imag = module.random.rand(rows, cols).astype(module.float32)
    return (real + 1j * imag) / module.sqrt(2 * cols)


def ru_weights_ptrbf(module, rows, cols, i=0):
    """
    Initializes a complex matrix with elements sampled from a uniform distribution.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.
    i : int, optional
        The number of neurons in input (default is 0).

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) with elements sampled from a uniform distribution.
    """
    real = module.random.rand(rows, cols).astype(module.float32)
    imag = module.random.rand(rows, cols).astype(module.float32)
    return (real + 1j * imag) * module.sqrt(module.exp(2) / (4 * rows * cols))


def opt_ptrbf_gamma(module, rows, cols, i=0):
    """
    Initializes complex gamma weights with elements sampled from a uniform distribution.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.
    i : int, optional
        The number of neurons in input (default is 0).

    Returns
    -------
    array-like
        A complex gamma weight matrix of size (rows, cols) with elements sampled from a uniform distribution.
    """
    real = module.random.uniform(-module.sqrt(3/2), module.sqrt(3/2), (rows, cols))
    imag = 1j * module.random.uniform(-module.sqrt(3/2), module.sqrt(3/2), (rows, cols))
    return (real + imag) * module.sqrt(1 / cols)


def opt_ptrbf_weights(module, rows, cols, i=0):
    """
    Initializes complex weights with elements sampled from a uniform distribution and adjusted for optimization.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.
    i : int, optional
        The number of neurons in input (default is 0).

    Returns
    -------
    array-like
        A complex weight matrix of size (rows, cols) with elements sampled from a uniform distribution
        and adjusted for optimization.
    """
    real = module.random.uniform(-module.sqrt(3/2), module.sqrt(3/2), (rows, cols))
    real = real - module.mean(real)
    imag = 1j * module.random.uniform(-module.sqrt(3/2), module.sqrt(3/2), (rows, cols))
    imag = imag - module.mean(imag)
    return (real + imag) * module.sqrt((5 * i) / (12 * rows * cols * module.exp(-2))) - module.mean((real + imag) * module.sqrt((5 * i) / (12 * rows * cols * module.exp(-2))), axis=0)


def opt_conv_ptrbf_weights(module, rows, cols, i=0):
    """
    Initializes complex convolutional weights with elements sampled from a uniform distribution and adjusted for optimization.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.
    i : int, optional
        The number of neurons in input (default is 0).

    Returns
    -------
    array-like
        A complex convolutional weight matrix of size (rows, cols) with elements sampled from a uniform distribution
        and adjusted for optimization.
    """
    real = module.random.uniform(-module.sqrt(3/2), module.sqrt(3/2), (rows, cols))
    real = real - module.mean(real)
    imag = 1j * module.random.uniform(-module.sqrt(3/2), module.sqrt(3/2), (rows, cols))
    imag = imag - module.mean(imag)
    return (real + imag) * module.sqrt((5 * i) / (12 * rows * cols * module.exp(-2))) / 100


def opt_crbf_gamma(module, rows, cols, i=0):
    """
    Initializes complex gamma weights for CRBF (Complex Radial Basis Function) with elements sampled from a uniform distribution.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.
    i : int, optional
        The number of neurons in input (default is 0).

    Returns
    -------
    array-like
        A complex gamma weight matrix of size (rows, cols) for CRBF with elements sampled from a uniform distribution.
    """
    real = module.random.uniform(-module.sqrt(3/2), module.sqrt(3/2), (rows, cols))
    imag = 1j * module.random.uniform(-module.sqrt(3/2), module.sqrt(3/2), (rows, cols))
    return (real + imag) * module.sqrt(1 / (2 * cols))


def opt_crbf_weights(module, rows, cols, i=0):
    """
    Initializes complex weights for CRBF (Complex Radial Basis Function) with elements sampled from a uniform distribution and adjusted for optimization.

    Parameters
    ----------
    module : module        
        CuPy/Numpy module.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.
    i : int, optional
        The number of neurons in input (default is 0).

    Returns
    -------
    array-like
        A complex weight matrix of size (rows, cols) for CRBF with elements sampled from a uniform distribution
        and adjusted for optimization.
    """
    real = module.random.uniform(-module.sqrt(3/2), module.sqrt(3/2), (rows, cols))
    real = real - module.mean(real)
    imag = 1j * module.random.uniform(-module.sqrt(3/2), module.sqrt(3/2), (rows, cols))
    imag = imag - module.mean(imag)
    return (real + imag) * module.sqrt((2 * 5 * i) / (12 * rows * cols * module.exp(-2))) - module.mean((real + imag) * module.sqrt((2 * 5 * i) / (12 * rows * cols * module.exp(-2))), axis=0)
