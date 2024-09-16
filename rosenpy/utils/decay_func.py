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
This file contains several decay functions used for adjusting learning rates in training processes.
"""

def none_decay(learning_rate, epoch, decay_rate, decay_steps=1):
    """
    Return the original learning rate without any decay.

    Parameters
    ----------
    learning_rate : float
        The initial learning rate.
    epoch : int
        The current epoch of the training process.
    decay_rate : float
        The decay rate (not used in this function).
    decay_steps : int, optional
        The number of steps at which to decay the learning rate (not used in this function). The default is 1.

    Returns
    -------
    float
        The original learning rate.
    """
    return learning_rate


def time_based_decay(learning_rate, epoch, decay_rate, decay_steps=1):
    """
    Apply time-based decay to the learning rate.

    Parameters
    ----------
    learning_rate : float
        The initial learning rate.
    epoch : int
        The current epoch of the training process.
    decay_rate : float
        The decay rate that controls the decay speed.
    decay_steps : int, optional
        The number of steps at which to decay the learning rate (default is 1).

    Returns
    -------
    float
        The decayed learning rate based on the epoch and decay rate.
    """
    return learning_rate / (1.0 + decay_rate * epoch)


def exponential_decay(learning_rate, epoch, decay_rate, decay_steps=1):
    """
    Apply exponential decay to the learning rate.

    Parameters
    ----------
    learning_rate : float
        The initial learning rate.
    epoch : int
        The current epoch of the training process.
    decay_rate : float
        The decay rate that controls the decay speed.
    decay_steps : int, optional
        The number of steps at which to decay the learning rate (default is 1).

    Returns
    -------
    float
        The decayed learning rate based on the epoch and decay rate.
    """
    return learning_rate * decay_rate ** epoch


def staircase(learning_rate, epoch, decay_rate, decay_steps=1):
    """
    Apply staircase decay to the learning rate.

    Parameters
    ----------
    learning_rate : float
        The initial learning rate.
    epoch : int
        The current epoch of the training process.
    decay_rate : float
        The decay rate that controls the decay speed.
    decay_steps : int, optional
        The number of steps at which to decay the learning rate (default is 1).

    Returns
    -------
    float
        The decayed learning rate based on the epoch and decay rate, using 
        the staircase decay formula.
    """
    return learning_rate * decay_rate ** (epoch // decay_steps)


