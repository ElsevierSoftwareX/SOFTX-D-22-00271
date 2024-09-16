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
along with RosenPy.  If not, see <http://www.gnu.org/licenses/>.
"""

from rosenpy.utils import act_func, init_func, decay_func


class Layer:
    """
    Specification for a layer to be passed to the Neural Network during construction. 
    This includes a variety of parameters to configure each layer based on its activation type.
    """
    
    def __init__(self, ishape, neurons, oshape=0, weights_initializer=init_func.random_normal, 
                 bias_initializer=init_func.random_normal, gamma_initializer=init_func.rbf_default, 
                 sigma_initializer=init_func.ones, activation=act_func.tanh, reg_strength=0.0, 
                 lambda_init=0.1, weights_rate=0.001, biases_rate=0.001, gamma_rate=0.0, sigma_rate=0.0, cvnn=1, 
                 lr_decay_method=decay_func.none_decay, lr_decay_rate=0.0, lr_decay_steps=1, 
                 kernel_initializer=init_func.opt_ptrbf_weights, kernel_size=3, module=None, category=1, 
                 layer_type="Fully"):
        """ 
        Initializes the Layer class with the specified parameters.
        
        Parameters
        ----------
        ishape : int
            The number of neurons in the first layer (the number of input features).  
        neurons : int
            The number of neurons in the hidden layer. 
        oshape : int
            Output shape, used for RBF networks.
        weights_initializer : function
            Function to initialize weights.
        bias_initializer : function
            Function to initialize biases.
        gamma_initializer : function, optional
            Function to initialize gamma (RBF networks).
        sigma_initializer : function, optional
            Function to initialize sigma (RBF networks).
        activation : function
            Activation function for the layer.
        reg_strength : float, optional
            Regularization strength, default is 0.0.
        lambda_init : float, optional
            Initial regularization factor strength.
        weights_rate : float, optional
            Learning rate for weights.
        biases_rate : float, optional
            Learning rate for biases.
        gamma_rate : float, optional
            Learning rate for gamma (RBF networks).
        sigma_rate : float, optional
            Learning rate for sigma (RBF networks).
        cvnn : int
            Complex neural network type.
        lr_decay_method : function
            Learning rate decay method.
        lr_decay_rate : float
            Learning rate decay rate.
        lr_decay_steps : int
            Steps for learning rate decay.
        kernel_initializer : function
            Function to initialize convolutional kernels.
        kernel_size : int
            Size of the kernel for convolutional layers.
        module : str
            CuPy/Numpy module, set at NeuralNetwork initialization.
        category : int
            Type of convolution: transient (1) or steady-state (0).
        layer_type : str
            Layer type: fully connected ("Fully") or convolutional ("Conv").
        """
        self.input = None
        self.reg_strength = reg_strength
        self.lambda_init = lambda_init
        self._activ_in, self._activ_out = None, None
        self.lr_decay_method = lr_decay_method  
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self.neurons = neurons
        self.oshape = oshape
        self.seuc = None
        self.phi = None
        self.kern = None
        self.layer_type = layer_type
        
        if cvnn == 1:
            self.learning_rates = [weights_rate, biases_rate]
            
            self.weights = weights_initializer(module, ishape, neurons)
            self.biases = bias_initializer(module, 1, neurons)
            self.activation = activation
            self.ut = self.mt = self.vt = [init_func.zeros(module, ishape, neurons), 
                                           init_func.zeros(module, 1, neurons)]
        elif cvnn == 2:
            self.learning_rates = [weights_rate, biases_rate, gamma_rate, sigma_rate]
            self.weights = weights_initializer(module, neurons, oshape, i=ishape)
            self.biases = bias_initializer(module, 1, oshape)
            self.gamma = gamma_initializer(module, neurons, ishape) 
            self.sigma = sigma_initializer(module, 1, neurons)
            self.ut = self.mt = self.vt = [init_func.zeros(module, neurons, oshape), 
                                           init_func.zeros(module, 1, oshape), 
                                           init_func.zeros(module, 1, neurons), 
                                           init_func.zeros(module, neurons, ishape)]
        elif cvnn == 3:
            self.learning_rates = [weights_rate, biases_rate, gamma_rate, sigma_rate]
            self.weights = weights_initializer(module, neurons, oshape)
            self.biases = bias_initializer(module, 1, oshape)
            self.gamma = gamma_initializer(module, neurons, ishape) 
            self.sigma = sigma_initializer(module, neurons, ishape) 
            self.ut = self.mt = self.vt = [init_func.zeros(module, neurons, oshape), 
                                           init_func.zeros(module, 1, oshape), 
                                           init_func.zeros(module, neurons, ishape), 
                                           init_func.zeros(module, neurons, ishape)]

        elif cvnn == 4 and self.layer_type == "Fully":
            self.learning_rates = [weights_rate, biases_rate, gamma_rate, sigma_rate]
            self.weights = weights_initializer(module, neurons, oshape, i=ishape)
            self.biases = bias_initializer(module, 1, oshape)
            self.gamma = gamma_initializer(module, neurons, ishape)
            self.sigma = sigma_initializer(module, 1, neurons)
            self.ut = self.mt = self.vt = [init_func.zeros(module, neurons, oshape), 
                                           init_func.zeros(module, 1, oshape), 
                                           init_func.zeros(module, 1, neurons), 
                                           init_func.zeros(module, neurons, ishape)]
        elif cvnn == 4 and self.layer_type == "Conv":
            self.category = category
            self.oshape = kernel_size + neurons - 1 if self.category == 1 else kernel_size - neurons + 1 if kernel_size > neurons else kernel_size
            self.learning_rates = [weights_rate, biases_rate, gamma_rate, sigma_rate]
            self.weights = weights_initializer(module, 1, kernel_size, i=ishape)
            self.biases = bias_initializer(module, 1, kernel_size + neurons - 1 if self.category == 1 else kernel_size - neurons + 1 if kernel_size > neurons else kernel_size)
            self.gamma = gamma_initializer(module, neurons, ishape)
            self.sigma = sigma_initializer(module, 1, neurons)
            self.kernel_size = kernel_size 
            self.ut = self.mt = self.vt = [init_func.zeros(module, 1, kernel_size), 
                                           init_func.zeros(module, 1, kernel_size + neurons - 1 if self.category == 1 else kernel_size - neurons + 1 if kernel_size > neurons else kernel_size), 
                                           init_func.zeros(module, 1, neurons), 
                                           init_func.zeros(module, neurons, ishape)]
            self.C = None
