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

from rosenpy.utils import reg_func, init_func, act_func, decay_func
from .rp_layer import Layer
from .rp_nn import NeuralNetwork

class CVFFNN(NeuralNetwork): 
    """
    Complex Valued FeedForward Neural Network (CVFFNN) class.
    
    This class handles feedforward, backpropagation, and layer addition operations 
    for a complex-valued neural network.
    """

    def feedforward(self, input_data):
        """
        Perform the feedforward operation on the neural network.
        
        Parameters
        ----------
        input_data : numpy.array or cupy.array
            The input data to feed into the network.
        
        Returns
        -------
        numpy.array or cupy.array
            The output of the final layer after feedforward.
        """
        if self.gpu_enable:
            return self._feedforward_gpu(input_data)
        return self._feedforward_cpu(input_data)

    def _feedforward_gpu(self, x):
        """
        Perform feedforward using GPU acceleration.
        
        Parameters
        ----------
        x : cupy.array
            The input data for the feedforward operation.
        
        Returns
        -------
        cupy.array
            The output of the final layer after feedforward.
        """
        layers = self.layers
        layers[0].input = x
        
        for i, layer in enumerate(layers):
            layer._activ_in = self.xp.dot(layer.input, layer.weights) + layer.biases
            layer._activ_out = layer.activation(self.xp, layer._activ_in, derivative=False)
            if i < len(layers) - 1:
                layers[i + 1].input = layer._activ_out
        
        return layers[-1]._activ_out

    def _feedforward_cpu(self, x):
        """
        Perform feedforward using CPU.
        
        Parameters
        ----------
        x : numpy.array
            The input data for the feedforward operation.
        
        Returns
        -------
        numpy.array
            The output of the final layer after feedforward.
        """
        self.layers[0].input = x
        self.layers[0]._activ_in = self.xp.dot(self.layers[0].input, self.layers[0].weights) + self.layers[0].biases
        self.layers[0]._activ_out = self.layers[0].activation(self.xp, self.layers[0]._activ_in, derivative=False)
        
        for i in range(1, len(self.layers)):
            self.layers[i].input = self.layers[i - 1]._activ_out
            self.layers[i]._activ_in = self.xp.dot(self.layers[i].input, self.layers[i].weights) + self.layers[i].biases
            self.layers[i]._activ_out = self.layers[i].activation(self.xp, self.layers[i]._activ_in, derivative=False)
        
        return self.layers[-1]._activ_out

    def backprop(self, y, y_pred, epoch):
        """
        Perform the backpropagation operation on the neural network.
        
        Parameters
        ----------
        y : numpy.array or cupy.array
            The true target values.
        y_pred : numpy.array or cupy.array
            The predicted values from the network.
        epoch : int
            The current training epoch.
        """
        if self.gpu_enable:
            return self._backprop_gpu(y, y_pred, epoch)
        return self._backprop_cpu(y, y_pred, epoch)

    def _backprop_gpu(self, y, y_pred, epoch):
        """
        Perform backpropagation using GPU acceleration.
        
        Parameters
        ----------
        y : cupy.array
            The true target values.
        y_pred : cupy.array
            The predicted values from the network.
        epoch : int
            The current training epoch.
        """
        e = y - y_pred
        delta_dir, aux_w = None, 0
        
        for layer in reversed(self.layers):
            deriv = layer.activation(self.xp, layer._activ_in, derivative=True)
            if delta_dir is not None:
                delta_dir = self.xp.multiply(self.xp.dot(delta_dir, self.xp.conj(aux_w.T)), self.xp.conj(deriv))
            else:
                delta_dir = self.xp.multiply(self.xp.conj(deriv), e)
            
            aux_w = layer.weights
            reg_l2 = reg_func.l2_regularization(self.xp, layer.lambda_init, layer.reg_strength, epoch)
            
            grad_w = self.xp.dot(self.xp.conj(layer.input.T), delta_dir) - (reg_l2 if layer.reg_strength else 0) * layer.weights
            grad_b = self.xp.mean(delta_dir, axis=0) - (reg_l2 if layer.reg_strength else 0) * layer.biases
            
            layer.weights, layer.biases, layer.mt, layer.vt, layer.ut = self.optimizer.update_parameters(
                [layer.weights, layer.biases], 
                [grad_w, grad_b], 
                layer.learning_rates, 
                epoch, layer.mt, layer.vt, layer.ut
            )

    def _backprop_cpu(self, y, y_pred, epoch):
        """
        Perform backpropagation using CPU.
        
        Parameters
        ----------
        y : numpy.array
            The true target values.
        y_pred : numpy.array
            The predicted values from the network.
        epoch : int
            The current training epoch.
        """
        e = y - y_pred
        delta_dir, aux_w = None, 0
        
        for layer in reversed(self.layers):
            deriv = layer.activation(self.xp, layer._activ_in, derivative=True)
            if delta_dir is not None:
                delta_dir = self.xp.multiply(self.xp.conj(deriv), self.xp.dot(delta_dir, self.xp.conjugate(aux_w.T)))
            else:
                delta_dir = self.xp.multiply(self.xp.conj(deriv), e)
            
            aux_w = layer.weights
            reg_l2 = reg_func.l2_regularization(self.xp, layer.lambda_init, layer.reg_strength, epoch)
            
            grad_w = self.xp.dot(self.xp.conj(layer.input.T), delta_dir) - (reg_l2 if layer.reg_strength else 0) * layer.weights
            grad_b = self.xp.divide(sum(delta_dir), delta_dir.shape[0]) - (reg_l2 if layer.reg_strength else 0) * layer.biases
            
            layer.weights, layer.biases, layer.mt, layer.vt, layer.ut = self.optimizer.update_parameters(
                [layer.weights, layer.biases], 
                [grad_w, grad_b], 
                layer.learning_rates, 
                epoch, layer.mt, layer.vt, layer.ut
            )

    def add_layer(self, neurons, ishape=0, weights_initializer=init_func.random_normal, 
                  bias_initializer=init_func.random_normal, activation=act_func.tanh,
                  weights_rate=0.001, biases_rate=0.001, 
                  reg_strength=0.0, lambda_init=0.1, lr_decay_method=decay_func.none_decay,  
                  lr_decay_rate=0.0, lr_decay_steps=1, module=None):
        """
        Add a layer to the neural network.
        
        Parameters
        ----------
        neurons : int
            The number of neurons in the layer.
        ishape : int, optional
            The input shape for the layer. Defaults to 0.
        weights_initializer : function, optional
            Function to initialize the weights. Defaults to random_normal.
        bias_initializer : function, optional
            Function to initialize the biases. Defaults to random_normal.
        activation : function, optional
            Activation function to use. Defaults to tanh.
        weights_rate : float, optional
            The learning rate for the weights. Defaults to 0.001.
        biases_rate : float, optional
            The learning rate for the biases. Defaults to 0.001.
        reg_strength : float, optional
            The regularization strength. Defaults to 0.0.
        lambda_init : float, optional
            The initial lambda for regularization. Defaults to 0.1.
        lr_decay_method : function, optional
            Method for decaying the learning rate. Defaults to none_decay.
        lr_decay_rate : float, optional
            The rate at which the learning rate decays. Defaults to 0.0.
        lr_decay_steps : int, optional
            The number of steps after which the learning rate decays. Defaults to 1.
        module : object, optional
            The module (e.g., NumPy or CuPy) to be used for computation. Defaults to None.
        """
        self.layers.append(Layer(
            ishape if not self.layers else self.layers[-1].neurons, neurons,
            weights_initializer=weights_initializer, 
            bias_initializer=bias_initializer, 
            activation=activation, 
            weights_rate=weights_rate, 
            biases_rate=biases_rate, 
            reg_strength=reg_strength, 
            lambda_init=lambda_init, 
            cvnn=1,
            lr_decay_method=lr_decay_method,  
            lr_decay_rate=lr_decay_rate, 
            lr_decay_steps=lr_decay_steps,
            module=self.xp
        ))
