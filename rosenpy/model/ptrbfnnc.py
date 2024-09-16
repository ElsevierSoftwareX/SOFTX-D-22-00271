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
from . import rp_optimizer as opt

class PTRBFNN(NeuralNetwork):
    """
    Specification for the Deep Phase Transmittance Radial Basis Function Neural Network 
    to be passed to the model in construction.
    This includes the feedforward, backpropagation, and adding layer methods specifics.
    
    This class derives from NeuralNetwork class.
    """

    def _matrix_c(self, phi, weights, trans_type):
        """
        Generates the coupling matrix C responsible for converting the linear combination 
        from the fully connected operation into a convolutional operation.

        Parameters:
        -----------
        phi : array-like
            Array containing the values of the basis functions.
        weights : array-like
            Array containing the values of the weights.
        trans_type : int
            Type of transformation (Transient and steady-state: 1; or steady-state: 0).

        Returns:
        --------
        array-like
            The coupling matrix C.
        """
        xp = self.xp  # Use cupy or numpy depending on the backend
        c_list = []

        if trans_type == 1:  # Transient and steady-state
            for phi_row in range(phi.shape[0]):
                m = phi.shape[1]
                n = weights.shape[1]
                c_matrix = xp.zeros((m + n - 1, n), dtype=phi.dtype)
                for row in range(m + n - 1):
                    for col in range(n):
                        if 0 <= row - col < m:
                            c_matrix[row, col] = phi[phi_row, row - col]
                c_list.append(c_matrix)
        else:  # steady-state
            for phi_row in range(phi.shape[0]):
                m = phi.shape[1]
                n = weights.shape[1]
                if m > n:
                    c_matrix = xp.zeros((m - n + 1, n), dtype=phi.dtype)
                    for row in range(m - n + 1):
                        for col in range(n):
                            c_matrix[row, col] = phi[phi_row, row + n - 1 - col]
                else:
                    c_matrix = xp.zeros((n - m + 1, n), dtype=phi.dtype)
                    for row in range(n - m + 1):
                        for col in range(n):
                            if 0 <= n - m + row - col - 1 < m:
                                c_matrix[row, col] = phi[phi_row, n - m + row - col - 1]
                c_list.append(c_matrix)

        return xp.stack(c_list)

    def _matrix_k(self, phi, weights, trans_type):
        """
        Generates the coupling matrix K responsible for transforming between 
        transient and steady-state or steady-state operations.

        Parameters:
        -----------
        phi : array-like
            Array containing the values of the basis functions.
        weights : array-like
            Array containing the values of the weights.
        trans_type : int
            Type of transformation (Transient and steady-state: 1; or steady-state: 0).

        Returns:
        --------
        array-like
            The coupling matrix K.
        """
        xp = self.xp  # Use cupy or numpy depending on the backend
        k_list = []

        if trans_type == 1:  # Transient and steady-state
            for phi_row in range(phi.shape[0]):
                m = phi.shape[1]
                n = weights.shape[1]
                k_matrix = xp.zeros((m, m + n - 1), dtype=xp.complex128)
                for row in range(n):
                    for col in range(m + n - 1):
                        if 0 <= col - row < n:
                            k_matrix[row, col] = weights[0, col - row]
                k_list.append(k_matrix)
        else:  # steady-state
            for phi_row in range(phi.shape[0]):
                m = phi.shape[1]
                n = weights.shape[1]
                if m > n:
                    k_matrix = xp.zeros((m, m - n + 1), dtype=phi.dtype)
                    for row in range(m):
                        for col in range(m - n + 1):
                            if 0 <= col - row + n - 1 < n:
                                k_matrix[row, col] = weights[0, col - row + n - 1]
                else:
                    k_matrix = xp.zeros((m, n - m + 1), dtype=phi.dtype)
                    for row in range(m):
                        for col in range(n - m + 1):
                            if 0 <= m - row + col - 1 < n:
                                k_matrix[row, col] = weights[0, m - row + col - 1]
                k_list.append(k_matrix)

        return xp.stack(k_list)

    def _fully_feedforward(self, y_pred, layer):
        """
        Performs the feedforward operation specific to a fully connected layer.

        Parameters:
        -----------
        y_pred : array-like
            The input data to be fed into the fully connected layer.
        layer : FullyConnectedLayer
            The fully connected layer object.

        Returns:
        --------
        array-like
            The output of the fully connected layer after the feedforward operation.
        """
        layer.kern = y_pred[:, self.xp.newaxis, :].repeat(layer.neurons, axis=1) - layer.gamma
        layer.seuc = (self.xp.sum(layer.kern.real ** 2, axis=2) / layer.sigma.real +
                      1j * self.xp.sum(layer.kern.imag ** 2, axis=2) / layer.sigma.imag)
        layer.phi = self.xp.exp(-layer.seuc.real) + 1j * self.xp.exp(-layer.seuc.imag)
        layer.activ_out = self.xp.dot(layer.phi, layer.weights) + layer.biases
        return layer.activ_out

    def _conv_feedforward_tp(self, x, layer):
        """
        Performs the feedforward operation specific to a convolutional layer.

        Parameters:
        -----------
        x : array-like
            The input data to be fed into the convolutional layer.
        layer : ConvLayer
            The convolutional layer object.

        Returns:
        --------
        array-like
            The output of the convolutional layer after the feedforward operation.
        """
        layer.input = self.xp.transpose(self.xp.tile(x, (layer.neurons, 1, 1)), axes=[1, 0, 2])
        layer.kern = layer.input - self.xp.tile(layer.gamma, (layer.input.shape[0], 1, 1))
        aux_r = self.xp.sum(layer.kern.real * layer.kern.real, axis=2)
        aux_i = self.xp.sum(layer.kern.imag * layer.kern.imag, axis=2)
        seuc_r = aux_r / layer.sigma.real
        seuc_i = aux_i / layer.sigma.imag
        layer.seuc = seuc_r + 1j * seuc_i
        layer.phi = self.xp.exp(-seuc_r) + 1j * self.xp.exp(-seuc_i)
        layer.C = self._matrix_c(layer.phi, layer.weights, layer.category)
        aux = self.xp.dot(layer.weights, self.xp.transpose(layer.C, (0, 2, 1)))
        layer.activ_out = self.xp.squeeze(aux) + layer.biases
        return layer.activ_out

    def feedforward(self, x):
        """
        Performs the feedforward operation on the neural network.

        Parameters:
        -----------
        x : array-like
            The input data to be fed into the neural network.

        Returns:
        --------
        array-like
            The output of the neural network after the feedforward operation.
        """
        conv_layer_found = False
        fully_connected_found = False

        for layer in self.layers:
            if layer.layer_type == "Conv":
                conv_layer_found = True
                x = self._conv_feedforward_tp(x, layer)
            elif layer.layer_type == "Fully":
                if not conv_layer_found:
                    fully_connected_found = True
                else:
                    if fully_connected_found:
                        raise ValueError("If there are convolutional layers, the last layer must be fully connected.")
                x = self._fully_feedforward(x, layer)
        return x

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

        Returns:
        --------
        array-like
            The gradients of the loss function with respect to the network parameters.
        """
        last = True
        aux_k = aux_r = aux_i = 0

        for layer in reversed(self.layers):
            if layer.layer_type == "Conv":
                aux_k, last, aux_r, aux_i = self._conv_backprop_tp(y, y_pred, epoch, layer, aux_k, last, aux_r, aux_i)
            elif layer.layer_type == "Fully":
                aux_k, last, aux_r, aux_i = self._fully_backprop(y, y_pred, epoch, layer, aux_k, last, aux_r, aux_i)

    def _conv_backprop_tp(self, y, y_pred, epoch, layer, aux_k, last, aux_r, aux_i):
        """
        Performs the backpropagation operation specific to a convolutional layer.

        Parameters:
        -----------
        y : array-like
            The true labels or target values.
        y_pred : array-like
            The predicted values from the convolutional layer.
        epoch : int
            The current epoch number.
        layer : ConvLayer
            The convolutional layer object.
        aux_k : array-like
            A kernel from the previous layer, which is obtained by subtracting the input by the gamma.
        last : bool
            Flag indicating if the current layer is the last layer in the network.
        aux_r : array-like
            Array containing the real part resulting from the multiplication of epsilon by phi under sigma.
        aux_i : array-like
            Array containing the imaginary part resulting from the multiplication of epsilon by phi under sigma.

        Returns:
        --------
        tuple
            A tuple containing the values to be used in the calculations of the following layers.
        """
        psi = -self.xp.sum(self.xp.matmul(self.xp.transpose(aux_k.real, (0, 2, 1)), aux_r[:, :, self.xp.newaxis]) +
                           1j * self.xp.matmul(self.xp.transpose(aux_k.imag, (0, 2, 1)), aux_i[:, :, self.xp.newaxis]), axis=2)
        aux_k = layer.kern

        k_matrix = self._matrix_k(layer.phi, layer.weights, layer.category)
        epsilon = self.xp.einsum('ij,ikj->ik', psi, self.xp.conj(k_matrix))

        psi_expanded = self.xp.transpose(self.xp.expand_dims(psi, axis=-1), axes=[1, 0, 2])
        beta_r = layer.phi.real / layer.sigma.real
        beta_i = layer.phi.imag / layer.sigma.imag
        aux_r = epsilon.real * beta_r
        aux_i = epsilon.imag * beta_i

        reg_l2 = reg_func.l2_regularization(self.xp, layer.lambda_init, layer.reg_strength, epoch)

        grad_w = (self.xp.tensordot(psi_expanded, self.xp.conj(layer.C), axes=([0, 1], [1, 0])) / layer.C.shape[0] -
                  (reg_l2 if layer.reg_strength else 0) * layer.weights)
        grad_b = self.xp.mean(psi, axis=0) - (reg_l2 if layer.reg_strength else 0) * layer.biases

        s_a = self.xp.multiply(aux_r, layer.seuc.real) + 1j * self.xp.multiply(aux_i, layer.seuc.imag)
        grad_s = self.xp.mean(s_a, axis=0) - (reg_l2 if layer.reg_strength else 0) * layer.sigma

        g_a = (self.xp.multiply(aux_r[:, :, self.xp.newaxis], layer.kern.real) +
               1j * self.xp.multiply(aux_i[:, :, self.xp.newaxis], layer.kern.imag))
        grad_g = self.xp.mean(g_a, axis=0) - (reg_l2 if layer.reg_strength else 0) * layer.gamma

        layer.weights, layer.biases, layer.sigma, layer.gamma, layer.mt, layer.vt, layer.ut = self.optimizer.update_parameters(
            [layer.weights, layer.biases, layer.sigma, layer.gamma],
            [grad_w, grad_b, grad_s, grad_g],
            layer.learning_rates,
            epoch, layer.mt, layer.vt, layer.ut
        )

        layer.sigma = self.xp.maximum(layer.sigma, 0.0001)
        return aux_k, last, aux_r, aux_i

    def _fully_backprop(self, y, y_pred, epoch, layer, aux_k, last, aux_r, aux_i):
        """
        Performs the backpropagation operation specific to a fully connected layer.

        Parameters:
        -----------
        y : array-like
            The true labels or target values.
        y_pred : array-like
            The predicted values from the fully connected layer.
        epoch : int
            The current epoch number.
        layer : FullyConnectedLayer
            The fully connected layer object.
        aux_k : array-like
            A kernel from the previous layer, which is obtained by subtracting the input by the gamma.
        last : bool
            Flag indicating if the current layer is the last layer in the network.
        aux_r : array-like
            Array containing the real part resulting from the multiplication of epsilon by phi under sigma.
        aux_i : array-like
            Array containing the imaginary part resulting from the multiplication of epsilon by phi under sigma.

        Returns:
        --------
        tuple
            A tuple containing the values to be used in the calculations of the following layers.
        """
        error = y - y_pred
        psi = error if last else -self.xp.sum(self.xp.matmul(self.xp.transpose(aux_k.real, (0, 2, 1)), aux_r[:, :, self.xp.newaxis]) +
                                              1j * self.xp.matmul(self.xp.transpose(aux_k.imag, (0, 2, 1)), aux_i[:, :, self.xp.newaxis]), axis=2)
        last = False
        aux_k = layer.kern

        epsilon = self.xp.dot(psi, self.xp.conj(layer.weights.T))
        beta_r = layer.phi.real / layer.sigma.real
        beta_i = layer.phi.imag / layer.sigma.imag
        aux_r = epsilon.real * beta_r
        aux_i = epsilon.imag * beta_i

        reg_l2 = reg_func.l2_regularization(self.xp, layer.lambda_init, layer.reg_strength, epoch)

        grad_w = self.xp.dot(self.xp.conj(layer.phi.T), psi) - (reg_l2 if layer.reg_strength else 0) * layer.weights
        grad_b = self.xp.mean(psi, axis=0) - (reg_l2 if layer.reg_strength else 0) * layer.biases

        s_a = self.xp.multiply(aux_r, layer.seuc.real) + 1j * self.xp.multiply(aux_i, layer.seuc.imag)
        grad_s = self.xp.mean(s_a, axis=0) - (reg_l2 if layer.reg_strength else 0) * layer.sigma

        g_a = (self.xp.multiply(aux_r[:, :, self.xp.newaxis], layer.kern.real) +
               1j * self.xp.multiply(aux_i[:, :, self.xp.newaxis], layer.kern.imag))
        grad_g = self.xp.mean(g_a, axis=0) - (reg_l2 if layer.reg_strength else 0) * layer.gamma

        layer.weights, layer.biases, layer.sigma, layer.gamma, layer.mt, layer.vt, layer.ut = self.optimizer.update_parameters(
            [layer.weights, layer.biases, layer.sigma, layer.gamma],
            [grad_w, grad_b, grad_s, grad_g],
            layer.learning_rates,
            epoch, layer.mt, layer.vt, layer.ut
        )

        layer.sigma = self.xp.maximum(layer.sigma, 0.0001)
        return aux_k, last, aux_r, aux_i

    def normalize_data(self, input_data, mean, std_dev):
        """
        Normalize the input data.

        Args:
            input_data (cupy/numpy.ndarray): Input data to be normalized.

        Returns:
            cupy/numpy.ndarray: Normalized input data.
        """
        return ((input_data - mean) / std_dev) * (1 / self.xp.sqrt(input_data.shape[1]))

    def denormalize_outputs(self, normalized_output_data, mean, std_dev):
        """
        Denormalize the output data.

        Args:
            normalized_output_data (cupy/numpy.ndarray): Normalized output data to be denormalized.
            
        Returns:
            cupy/numpy.ndarray: Denormalized output data.
        """
        return (normalized_output_data * std_dev) / (1 / self.xp.sqrt(normalized_output_data.shape[1])) + mean

    def add_layer(self, neurons, ishape=0, oshape=0, weights_initializer=init_func.opt_ptrbf_weights,
              bias_initializer=init_func.zeros, sigma_initializer=init_func.ones, gamma_initializer=init_func.opt_ptrbf_gamma,
              reg_strength=0.0, lambda_init=0.1, weights_rate=0.001, biases_rate=0.001, gamma_rate=0.01, sigma_rate=0.01,
              lr_decay_method=decay_func.none_decay, lr_decay_rate=0.0, lr_decay_steps=1,
              kernel_initializer=init_func.opt_ptrbf_gamma, kernel_size=3,
              module=None, category=1,
              layer_type="Fully"):
        """
        Adds a layer to the neural network.
    
        This method is responsible for appending a new layer to the neural network structure. 
        The layer can be fully connected or convolutional, depending on the parameters provided.
    
        Parameters
        ----------
        neurons : int
            The number of neurons in the hidden layer. If `ishape` is different from zero 
            and this is the first layer of the model, `neurons` represents the number of 
            neurons in the first layer (i.e., the number of input features).
        ishape : int, optional
            The number of neurons in the first layer (i.e., the number of input features). Default is 0.
        oshape : int, optional
            The number of output neurons (shape of the output). If not provided, defaults to the number of neurons. Default is 0.
        weights_initializer : function, optional
            The function used to initialize the layer's weights. Default is `init_func.opt_ptrbf_weights`.
        bias_initializer : function, optional
            The function used to initialize the layer's biases. Default is `init_func.zeros`.
        sigma_initializer : function, optional
            The function used to initialize the `sigma` parameter. Default is `init_func.ones`.
        gamma_initializer : function, optional
            The function used to initialize the `gamma` parameter. Default is `init_func.opt_ptrbf_gamma`.
        reg_strength : float, optional
            The strength of L2 regularization applied to the layer. Default is 0.0 (no regularization).
        lambda_init : float, optional
            The initial value for the regularization term. Default is 0.1.
        weights_rate : float, optional
            The learning rate applied to the weights during training. Default is 0.001.
        biases_rate : float, optional
            The learning rate applied to the biases during training. Default is 0.001.
        gamma_rate : float, optional
            The learning rate applied to the `gamma` parameter during training. Default is 0.01.
        sigma_rate : float, optional
            The learning rate applied to the `sigma` parameter during training. Default is 0.01.
        lr_decay_method : function, optional
            The method used for decaying the learning rate over time. Default is `decay_func.none_decay`.
        lr_decay_rate : float, optional
            The rate at which the learning rate decays. Default is 0.0 (no decay).
        lr_decay_steps : int, optional
            The number of steps after which the learning rate decays. Default is 1.
        kernel_initializer : function, optional
            The function used to initialize the kernel for convolutional layers. Default is `init_func.opt_ptrbf_gamma`.
        kernel_size : int, optional
            The size of the convolutional kernel. Default is 3.
        module : object, optional
            The computation module used (e.g., NumPy or CuPy). If not provided, it is set during the initialization of the `NeuralNetwork` class. Default is None.
        category : int, optional
            The type of convolution: 1 for transient and steady-state, 0 for steady-state only. Default is 1.
        layer_type : str, optional
            The type of layer to add: "Fully" for fully connected layers, "Conv" for convolutional layers. Default is "Fully".
    
        Returns
        -------
        None
            This method does not return any value; it modifies the network structure by appending a new layer.
    
        Notes
        -----
        The layer is added to the `self.layers` list, which is a sequence of layers in the neural network.
        The parameters provided, such as initialization methods and learning rates, are specific to each layer.
        """
        self.layers.append(Layer(
            ishape if not len(self.layers) else self.layers[-1].oshape, neurons, neurons if not oshape else oshape,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            sigma_initializer=sigma_initializer,
            gamma_initializer=gamma_initializer,
            reg_strength=reg_strength,
            lambda_init=lambda_init,
            weights_rate=weights_rate,
            biases_rate=biases_rate,
            sigma_rate=sigma_rate,
            gamma_rate=gamma_rate,
            cvnn=4,
            lr_decay_method=lr_decay_method,
            lr_decay_rate=lr_decay_rate,
            lr_decay_steps=lr_decay_steps,
            kernel_initializer=kernel_initializer,
            kernel_size=kernel_size,
            module=self.xp,
            category=category,
            layer_type=layer_type
        ))
