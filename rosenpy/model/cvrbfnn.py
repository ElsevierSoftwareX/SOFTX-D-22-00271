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


class CVRBFNN(NeuralNetwork):
    """
    Specification for the Complex Valued Radial Basis Function Neural Network.
    This includes the feedforward, backpropagation, and adding layer methods specifics.
    This class derives from NeuralNetwork class.
    """

    def feedforward(self, input_data):
        """
        Performs the feedforward operation on the neural network.

        Parameters:
        -----------
        input_data : array-like
            The input data to be fed into the neural network.

        Returns:
        --------
        array-like
            The output of the neural network after the feedforward operation.
        """
        if self.gpu_enable:
            return self._feedforward_gpu(input_data)
        return self._feedforward_cpu(input_data)

    def _feedforward_gpu(self, x):
        """
        Performs the feedforward operation using GPU.

        Parameters:
        -----------
        x : array-like
            Input data.

        Returns:
        --------
        array-like
            Output of the last layer.
        """
        x_gpu = self.xp.array(x)
        self.layers[0].input = self.xp.transpose(self.xp.tile(x, (self.layers[0].neurons, 1, 1)), axes=[1, 0, 2])
        self.layers[0].kern = self.layers[0].input - self.xp.tile(self.layers[0].gamma, (self.layers[0].input.shape[0], 1, 1))
        self.layers[0].seuc = self.xp.sum(self.xp.abs(self.layers[0].kern**2), axis=2) / self.layers[0].sigma
        self.layers[0].phi = self.xp.exp(-self.layers[0].seuc)
        self.layers[0].activ_out = self.xp.dot(self.layers[0].phi, self.layers[0].weights) + self.layers[0].biases

        for i in range(1, len(self.layers)):
            self.layers[i].input = self.xp.transpose(self.xp.tile(self.layers[i - 1].activ_out, (self.layers[i].neurons, 1, 1)), axes=[1, 0, 2])
            self.layers[i].kern = self.layers[i].input - self.xp.tile(self.layers[i].gamma, (self.layers[i].input.shape[0], 1, 1))
            self.layers[i].seuc = self.xp.sum(self.xp.abs(self.layers[i].kern**2), axis=2) / self.layers[i].sigma
            self.layers[i].phi = self.xp.exp(-self.layers[i].seuc)
            self.layers[i].activ_out = self.xp.dot(self.layers[i].phi, self.layers[i].weights) + self.layers[i].biases
        
        return self.layers[-1].activ_out.get()

    def _feedforward_cpu(self, x):
        """
        Performs the feedforward operation using CPU.

        Parameters:
        -----------
        x : array-like
            Input data.

        Returns:
        --------
        array-like
            Output of the last layer.
        """
        self.layers[0].input = self.xp.transpose(self.xp.tile(x, (self.layers[0].neurons, 1, 1)), axes=[1, 0, 2])
        self.layers[0].kern = self.layers[0].input - self.xp.tile(self.layers[0].gamma, (self.layers[0].input.shape[0], 1, 1))
        self.layers[0].seuc = self.xp.sum(self.xp.abs(self.layers[0].kern**2), axis=2) / self.layers[0].sigma
        self.layers[0].phi = self.xp.exp(-self.layers[0].seuc)
        self.layers[0].activ_out = self.xp.dot(self.layers[0].phi, self.layers[0].weights) + self.layers[0].biases

        for i in range(1, len(self.layers)):
            self.layers[i].input = self.xp.transpose(self.xp.tile(self.layers[i - 1].activ_out, (self.layers[i].neurons, 1, 1)), axes=[1, 0, 2])
            self.layers[i].kern = self.layers[i].input - self.xp.tile(self.layers[i].gamma, (self.layers[i].input.shape[0], 1, 1))
            self.layers[i].seuc = self.xp.sum(self.xp.abs(self.layers[i].kern**2), axis=2) / self.layers[i].sigma
            self.layers[i].phi = self.xp.exp(-self.layers[i].seuc)
            self.layers[i].activ_out = self.xp.dot(self.layers[i].phi, self.layers[i].weights) + self.layers[i].biases
        
        return self.layers[-1].activ_out

    def backprop(self, y, y_pred, epoch):
        """
        Performs the backpropagation operation on the neural network.

        Parameters:
        -----------
        y : array-like
            The true labels or target values.
        y_pred : array-like
            The predicted values from the neural network.
        epoch : int
            The current epoch number.
        """
        if self.gpu_enable:
            return self._backprop_gpu(y, y_pred, epoch)
        return self._backprop_cpu(y, y_pred, epoch)

    def _backprop_cpu(self, y, y_pred, epoch):
        """
        Performs the backpropagation operation using CPU.

        Parameters:
        -----------
        y : array-like
            Target values.
        y_pred : array-like
            Predicted values.
        epoch : int
            Current epoch number.
        """
        error = y - y_pred
        last = True
        aux_k = aux = 0

        for layer in reversed(self.layers):
            psi = error if last else -self.xp.sum(self.xp.matmul(self.xp.transpose(aux_k, (0, 2, 1)), aux[:, :, self.xp.newaxis]), axis=2)
            last = False
            aux_k = layer.kern
            epsilon = self.xp.dot(psi.real, layer.weights.real.T) + self.xp.dot(psi.imag, layer.weights.imag.T)
            beta = layer.phi / layer.sigma
            aux = self.xp.multiply(epsilon, beta)

            reg_l2 = reg_func.l2_regularization(self.xp, layer.lambda_init, layer.reg_strength, epoch)
            grad_w = self.xp.dot(layer.phi.T, psi) - (reg_l2 if layer.reg_strength else 0) * layer.weights
            grad_b = self.xp.divide(sum(psi), psi.shape[0]) - (reg_l2 if layer.reg_strength else 0) * layer.biases
            s_a = self.xp.multiply(aux, layer.seuc)
            grad_s = self.xp.divide(sum(s_a), s_a.shape[0]) - (reg_l2 if layer.reg_strength else 0) * layer.sigma
            g_a = self.xp.multiply(aux[:, :, self.xp.newaxis], layer.kern)
            grad_g = self.xp.divide(sum(g_a), g_a.shape[0]) - (reg_l2 if layer.reg_strength else 0) * layer.gamma

            layer.weights, layer.biases, layer.sigma, layer.gamma, layer.mt, layer.vt, layer.ut = self.optimizer.update_parameters(
                [layer.weights, layer.biases, layer.sigma, layer.gamma],
                [grad_w, grad_b, grad_s, grad_g],
                layer.learning_rates,
                epoch, layer.mt, layer.vt, layer.ut
            )

            layer.sigma = self.xp.where(layer.sigma.real > 0.0001, layer.sigma.real, 0.0001)

    def _backprop_gpu(self, y, y_pred, epoch):
        """
        Performs the backpropagation operation using GPU.

        Parameters:
        -----------
        y : array-like
            Target values.
        y_pred : array-like
            Predicted values.
        epoch : int
            Current epoch number.
        """
        error = y - y_pred
        last = True
        aux_k = aux = 0

        for layer in reversed(self.layers):
            psi = error if last else -self.xp.sum(self.xp.matmul(self.xp.transpose(aux_k, (0, 2, 1)), aux[:, :, self.xp.newaxis]), axis=2)
            last = False
            aux_k = layer.kern

            epsilon_real = self.xp.dot(psi.real, layer.weights.real.T)
            epsilon_imag = self.xp.dot(psi.imag, layer.weights.imag.T)
            epsilon_real_beta = epsilon_real * layer.phi / layer.sigma
            epsilon_imag_beta = epsilon_imag * layer.phi / layer.sigma

            reg_l2 = reg_func.l2_regularization(self.xp, layer.lambda_init, layer.reg_strength, epoch)

            grad_w = self.xp.dot(layer.phi.T, psi) - (reg_l2 if layer.reg_strength else 0) * layer.weights
            grad_b = self.xp.mean(psi) - (reg_l2 if layer.reg_strength else 0) * layer.biases
            s_a = self.xp.multiply(aux, layer.seuc)
            grad_s = self.xp.mean(s_a) - (reg_l2 if layer.reg_strength else 0) * layer.sigma
            g_a = self.xp.multiply(aux[:, :, self.xp.newaxis], layer.kern)
            grad_g = self.xp.mean(g_a) - (reg_l2 if layer.reg_strength else 0) * layer.gamma

            layer.weights, layer.biases, layer.sigma, layer.gamma, layer.mt, layer.vt, layer.ut = self.optimizer.update_parameters(
                [layer.weights, layer.biases, layer.sigma, layer.gamma],
                [grad_w, grad_b, grad_s, grad_g],
                layer.learning_rates,
                epoch, layer.mt, layer.vt, layer.ut
            )

            layer.sigma = self.xp.where(layer.sigma.real > 0.0001, layer.sigma.real, 0.0001)

    def normalize_data(self, input_data, mean, std_dev):
        """
        Normalize the input data.

        Parameters:
        -----------
        input_data : array-like
            Input data to be normalized.
        mean : float
            Mean value for normalization.
        std_dev : float
            Standard deviation for normalization.

        Returns:
        --------
        array-like
            Normalized input data.
        """
        return ((input_data - mean) / std_dev) * (1 / self.xp.sqrt(2 * input_data.shape[1]))

    def denormalize_outputs(self, normalized_output_data, mean, std_dev):
        """
        Denormalize the output data.

        Parameters:
        -----------
        normalized_output_data : array-like
            Normalized output data.
        mean : float
            Mean value for denormalization.
        std_dev : float
            Standard deviation for denormalization.

        Returns:
        --------
        array-like
            Denormalized output data.
        """
        return (normalized_output_data / (1 / self.xp.sqrt(2 * normalized_output_data.shape[1]))) * std_dev + mean

    def add_layer(self, neurons, ishape=0, oshape=0, weights_initializer=init_func.opt_crbf_weights, bias_initializer=init_func.zeros,
                  sigma_initializer=init_func.ones_real, gamma_initializer=init_func.opt_crbf_gamma, weights_rate=0.001, biases_rate=0.001,
                  gamma_rate=0.01, sigma_rate=0.01, reg_strength=0.0, lambda_init=0.1, lr_decay_method=decay_func.none_decay,
                  lr_decay_rate=0.0, lr_decay_steps=1, module=None):
        """
        Adds a layer to the neural network.

        Parameters:
        -----------
        neurons : int
            Number of neurons in the layer.
        ishape : int, optional
            Input shape for the layer.
        oshape : int, optional
            Output shape for the layer.
        weights_initializer : function, optional
            Function to initialize the weights.
        bias_initializer : function, optional
            Function to initialize the biases.
        sigma_initializer : function, optional
            Function to initialize sigma values.
        gamma_initializer : function, optional
            Function to initialize gamma values.
        weights_rate : float, optional
            Learning rate for weights.
        biases_rate : float, optional
            Learning rate for biases.
        gamma_rate : float, optional
            Learning rate for gamma.
        sigma_rate : float, optional
            Learning rate for sigma.
        reg_strength : float, optional
            Regularization strength.
        lambda_init : float, optional
            Initial lambda value for regularization.
        lr_decay_method : function, optional
            Learning rate decay method.
        lr_decay_rate : float, optional
            Learning rate decay rate.
        lr_decay_steps : int, optional
            Learning rate decay steps.
        module : module, optional
            Module for computation (e.g., numpy, cupy).

        Returns:
        --------
        None
        """
        self.layers.append(Layer(
            ishape if not len(self.layers) else self.layers[-1].oshape, neurons, neurons if not oshape else oshape,
            weights_initializer=weights_initializer, bias_initializer=bias_initializer,
            sigma_initializer=sigma_initializer, gamma_initializer=gamma_initializer,
            reg_strength=reg_strength, lambda_init=lambda_init, weights_rate=weights_rate, biases_rate=biases_rate,
            sigma_rate=sigma_rate, gamma_rate=gamma_rate, cvnn=2,
            lr_decay_method=lr_decay_method, lr_decay_rate=lr_decay_rate,
            lr_decay_steps=lr_decay_steps, module=self.xp
        ))
