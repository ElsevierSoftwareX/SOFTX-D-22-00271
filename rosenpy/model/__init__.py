# -*- coding: utf-8 -*-
"""**RosenPy: An Open Source Python Framework for Complex-Valued Neural Networks**.
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

# __init__.py for the 'model' submodule of the 'rosenpy' package
# This file serves as the initializer for the 'model' submodule and imports key classes and functions
# from various modules to make them available at the 'rosenpy.model' namespace.

# Importing different types of neural network architectures
from .cvffnn import CVFFNN  # Complex-Valued Feedforward Neural Network
from .scffnn import SCFFNN  # Split-Complex Feedforward Neural Network
from .cvrbfnn import CVRBFNN  # Complex-Valued Radial Basis Function Neural Network
from .fcrbfnn import FCRBFNN  # Fully Connected Radial Basis Function Neural Network
from .ptrbfnnc import PTRBFNN  # Parametric Transfer Radial Basis Function Neural Network (Convolutional variant)

# Importing the Layer class for defining neural network layers
from .rp_layer import Layer

# Importing the NeuralNetwork class, a base class for building neural network models
from .rp_nn import NeuralNetwork

# Importing optimizers for training neural networks
from .rp_optimizer import (
    Optimizer,  # Base class for optimizers
    GradientDescent,  # Standard Gradient Descent optimizer
    Adam,  # Adam optimizer for stochastic gradient descent
    CVAdam,  # Complex-Valued version of the Adam optimizer
    AMSGrad,  # AMSGrad, a variant of Adam with better convergence properties
    CVAMSGrad,  # Complex-Valued AMSGrad optimizer
    Nadam,  # Nesterov-accelerated Adaptive Moment Estimation (Nadam) optimizer
    CVNadam  # Complex-Valued Nadam optimizer
)

# Specifying the public API of this submodule using __all__
# These are the classes and functions that will be available for import from 'rosenpy.model'
__all__ = [
    'CVFFNN',  # Complex-Valued Feedforward Neural Network
    'CVRBFNN',  # Complex-Valued Radial Basis Function Neural Network
    'FCRBFNN',  # Fully Connected Radial Basis Function Neural Network
    'PTRBFNNConv',  # Parametric Transfer Radial Basis Function Neural Network (Convolutional variant)
    'Layer',  # Class for defining layers in the neural network
    'NeuralNetwork',  # Base class for building neural network models
    'Optimizer',  # Base class for optimizers
    'GradientDescent',  # Gradient Descent optimizer
    'Adam',  # Adam optimizer
    'CVAdam',  # Complex-Valued Adam optimizer
    'AMSGrad',  # AMSGrad optimizer
    'CVAMSGrad',  # Complex-Valued AMSGrad optimizer
    'Nadam',  # Nadam optimizer
    'CVNadam',  # Complex-Valued Nadam optimizer
    'SCFFNN'  # Split-Complex Feedforward Neural Network
]
