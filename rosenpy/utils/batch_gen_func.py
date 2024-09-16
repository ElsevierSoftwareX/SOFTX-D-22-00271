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

"""
This file contains the functions used to produce batches. The batches will be 
sequential or shuffled.
"""

def batch_sequential(module, x, y, batch_size=1):
    """
    Generates sequential batches of data for neural network training.

    Parameters
    ----------
    xp : module
        CuPy or NumPy module for array handling.
    x : array-like, shape (n_samples, n_inputs)
        Training vectors as real numbers, where n_samples is the number of
        samples and n_inputs is the number of input features.
    y : array-like, shape (n_samples, n_outputs)
        Target values representing the desired outputs.
    batch_size : int, optional
        Size of each batch. If batch_size equals 1, the algorithm will 
        perform Stochastic Gradient Descent (SGD). Default is 1.

    Returns
    -------
    x_batches : array-like, shape (n_batches, batch_size, n_inputs)
        Batches of training inputs.
    y_batches : array-like, shape (n_batches, batch_size, n_outputs)
        Batches of target outputs.
    """
    n_batches = (x.shape[0] + batch_size - 1) // batch_size
    x_batches, y_batches = [], []

    for i in range(n_batches):
        start, end = i * batch_size, (i + 1) * batch_size
        x_batches.append(x[start:end])
        y_batches.append(y[start:end])

    return module.array(x_batches), module.array(y_batches)

def batch_shuffle(module, x, y, batch_size=1):
    """
    Generates shuffled batches of data for neural network training.

    Parameters
    ----------
    xp : module
        CuPy or NumPy module for array handling.
    x : array-like, shape (n_samples, n_inputs)
        Training vectors as real numbers, where n_samples is the number of
        samples and n_inputs is the number of input features.
    y : array-like, shape (n_samples, n_outputs)
        Target values representing the desired outputs.
    batch_size : int, optional
        Size of each batch. If batch_size equals 1, the algorithm will 
        perform Stochastic Gradient Descent (SGD). Default is 1.

    Returns
    -------
    x_batches : array-like, shape (n_batches, batch_size, n_inputs)
        Batches of shuffled training inputs.
    y_batches : array-like, shape (n_batches, batch_size, n_outputs)
        Batches of shuffled target outputs.
    """
    shuffle_indices = xp.random.permutation(x.shape[0])
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    
    return batch_sequential(module, x_shuffled, y_shuffled, batch_size)
