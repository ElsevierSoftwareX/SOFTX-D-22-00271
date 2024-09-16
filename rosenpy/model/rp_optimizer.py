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

from rosenpy.utils import act_func, init_func

class Optimizer:
    """
    Base class for all optimizers used in the neural network.

    This class defines common parameters and methods that can be used
    by all derived optimizers.
    """
    def __init__(self, beta=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initializes the optimizer with default hyperparameters.

        Parameters:
        -----------
        beta : float, optional
            The value for the beta parameter. Default is 100.
        beta1 : float, optional
            The value for the beta1 parameter. Default is 0.9.
        beta2 : float, optional
            The value for the beta2 parameter. Default is 0.999.
        epsilon : float, optional
            A small constant added to prevent division by zero. Default is 1e-8.
        """
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.vt = None
        self.ut = None
        self.xp = None
        self.optimizer = None
    
    def set_module(self, xp):
        """
        Sets the backend module (NumPy or CuPy) for matrix operations.

        Parameters:
        -----------
        xp : module
            The backend module (NumPy or CuPy).
        """
        self.xp = xp
   
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        """
        Updates the parameters of the neural network based on the gradients.

        This is a placeholder method that should be implemented by subclasses.

        Parameters:
        -----------
        parameters : tuple
            The parameters of the neural network.
        gradients : tuple
            The gradients of the loss function with respect to the parameters.
        learning_rate : tuple
            The learning rates for updating the parameters.
        epoch : int
            The current epoch number.
        mt : tuple
            The first moment estimates.
        vt : tuple
            The second moment estimates.
        ut : tuple
            The third moment estimates.

        Returns:
        --------
        tuple
            The updated parameters along with the updated moment estimates.
        """
        raise NotImplementedError("Subclasses must implement update_parameters method.")

class GradientDescent(Optimizer):
    """
    Gradient Descent optimizer.

    This class implements the standard gradient descent optimization algorithm.
    """
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        """
        Updates the parameters using the gradient descent optimizer.

        Parameters:
        -----------
        parameters : tuple
            The parameters of the neural network.
        gradients : tuple
            The gradients of the loss function with respect to the parameters.
        learning_rate : tuple
            The learning rates for updating the parameters.
        epoch : int
            The current epoch number.
        mt : tuple
            The first moment estimates (not used in this optimizer).
        vt : tuple
            The second moment estimates (not used in this optimizer).
        ut : tuple
            The third moment estimates (not used in this optimizer).

        Returns:
        --------
        tuple
            The updated parameters.
        """
        return tuple(p + lr * g for p, g, lr in zip(parameters, gradients, learning_rate)) + (mt, vt, ut)

class Adam(Optimizer):
    """
    Adam optimizer.

    This class implements the Adam optimization algorithm, which is an adaptive 
    learning rate optimization algorithm.
    """
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        """
        Updates the parameters using the Adam optimizer.

        Parameters:
        -----------
        Same as the parent class.

        Returns:
        --------
        tuple
            The updated parameters along with the updated moment estimates.
        """
        updated_parameters, updated_mt, updated_vt = [], [], []
        for p, g, lr, m, v in zip(parameters, gradients, learning_rate, mt, vt):
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (self.xp.abs(g) ** 2)
            mc = m / (1 - self.beta1 ** epoch)
            vc = v / (1 - self.beta2 ** epoch)
            updated_parameters.append(p + lr * (mc / (self.xp.sqrt(vc) + self.epsilon)))
            updated_mt.append(m)
            updated_vt.append(v)
        return tuple(updated_parameters + [updated_mt, updated_vt, ut])

class CVAdam(Optimizer):
    """
    Complex-Valued Adam optimizer.

    This class implements the complex-valued version of the Adam optimization algorithm.
    """
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        """
        Updates the parameters using the complex-valued Adam optimizer.

        Parameters:
        -----------
        Same as the parent class.

        Returns:
        --------
        tuple
            The updated parameters along with the updated moment estimates.
        """
        updated_parameters, updated_mt, updated_vt = [], [], []
        for p, g, lr, m, v in zip(parameters, gradients, learning_rate, mt, vt):
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (self.xp.real(g) ** 2 + 1j * self.xp.imag(g) ** 2)
            mc = m / (1 - self.beta1 ** epoch)
            vc = v / (1 - self.beta2 ** epoch)
            r = self.xp.real(mc) / (self.xp.sqrt(self.xp.real(vc)) + self.epsilon)
            i = self.xp.imag(mc) / (self.xp.sqrt(self.xp.imag(vc)) + self.epsilon)
            updated_parameters.append(p + lr * (r + 1j * i))
            updated_mt.append(m)
            updated_vt.append(v)
        return tuple(updated_parameters + [updated_mt, updated_vt, ut])



class AMSGrad(Optimizer):
    """
    AMSGrad optimizer.

    This class implements the AMSGrad optimization algorithm, a variant of Adam that 
    improves convergence in certain cases by keeping track of the maximum past squared gradient.
    """
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        """
        Updates the parameters using the AMSGrad optimizer.

        Parameters:
        -----------
        Same as the parent class.

        Returns:
        --------
        tuple
            The updated parameters along with the updated moment estimates.
        """
        updated_parameters, updated_mt, updated_vt, updated_ut = [], [], [], []
        for p, g, lr, m, v, u in zip(parameters, gradients, learning_rate, mt, vt, ut):
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (self.xp.abs(g) ** 2)
            u = self.xp.maximum(u, v)
            updated_parameters.append(p + lr * (m / (self.xp.sqrt(u) + self.epsilon)))
            updated_mt.append(m)
            updated_vt.append(v)
            updated_ut.append(u)
        return tuple(updated_parameters + [updated_mt, updated_vt, updated_ut])

class SAMSGrad(Optimizer):
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        """
        Updates the parameters using the SAMSGrad optimizer.

        Parameters:
        -----------
        Same as the parent class.

        Returns:
        --------
        tuple
            The updated parameters along with the updated moment estimates.
        """
        updated_parameters, updated_mt, updated_vt, updated_ut = [], [], [], []
        for p, g, lr, m, v, u in zip(parameters, gradients, learning_rate, mt, vt, ut):
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (self.xp.abs(g) ** 2)
            u = self.xp.maximum(self.xp.abs(u), self.xp.abs(v))

            updated_params.append(p + lr * (m / ((1 / self.beta) * self.xp.log(1 + self.xp.exp(self.beta * self.xp.sqrt(u))))))
            updated_mt.append(m)
            updated_vt.append(v)
            updated_ut.append(u)
        return tuple(updated_parameters + [updated_mt, updated_vt, updated_ut])

class CVAMSGrad(Optimizer):
    """
        Updates the parameters using the complex-valued SAMSGrad optimizer.

        Parameters:
        -----------
        Same as the parent class.

        Returns:
        --------
        tuple
            The updated parameters along with the updated moment estimates.
        """
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        updated_parameters, updated_mt, updated_vt, updated_ut = [], [], [], []

        for p, g, lr, m, v, u in zip(parameters, gradients, learning_rate, mt, vt, ut):
           m = self.beta1 * m + (1 - self.beta1) * g
           v = self.beta2 * v + (1 - self.beta2) * (self.xp.real(g) ** 2 + 1j * self.xp.imag(g) ** 2)
           u = self.xp.maximum(self.xp.abs(self.xp.real(u)), self.xp.abs(self.xp.real(v))) + 1j * self.xp.maximum(self.xp.abs(self.xp.imag(u)), self.xp.abs(self.xp.imag(v)))
           
           real_part = self.xp.real(m) / (self.xp.sqrt(self.xp.real(u)) + self.epsilon)
           imag_part = self.xp.imag(m) / (self.xp.sqrt(self.xp.imag(u)) + self.epsilon)
           
           updated_parameters.append(p + lr * (real_part + 1j * imag_part))
           updated_mt.append(m)
           updated_vt.append(v)
           updated_ut.append(u)
        return tuple(updated_parameters + [updated_mt, updated_vt, updated_ut])  
    
class Adamax(Optimizer):
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        """
        Updates the parameters using the Adamax optimizer.

        Parameters:
        -----------
        Same as the parent class.

        Returns:
        --------
        tuple
            The updated parameters along with the updated moment estimates.
        """
        updated_parameters, updated_mt, updated_vt = [], [], []
    
        for p, g, lr, m, v in zip(parameters, gradients, learning_rate, mt, vt):
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.xp.maximum(self.beta2 * v, self.xp.abs(g))
            updated_parameters.append(p + (lr / (1 - self.beta1 ** epoch)) * m / (v + self.epsilon))
            updated_mt.append(m)
            updated_vt.append(v)
        
        return tuple(updated_parameters + [updated_mt, updated_vt, ut])


class CVAdamax(Optimizer):
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        """
        Updates the parameters using the complex-valued Adamax optimizer.

        Parameters:
        -----------
        Same as the parent class.

        Returns:
        --------
        tuple
            The updated parameters along with the updated moment estimates.
        """
        updated_parameters, updated_mt, updated_vt = [], [], []
        
        for p, g, lr, m, v in zip(parameters, gradients, learning_rate, mt, vt):
            m = self.beta1 * m + (1 - self.beta1) * g
            v_real = self.xp.maximum(self.beta2 * self.xp.real(v), self.xp.abs(self.xp.real(g)))
            v_imag = self.xp.maximum(self.beta2 * self.xp.imag(v), self.xp.abs(self.xp.imag(g)))
            v = v_real + 1j * v_imag
            
            real_part = self.xp.real(m) / (self.xp.real(v) + self.epsilon)
            imag_part = self.xp.imag(m) / (self.xp.imag(v) + self.epsilon)
            
            updated_parameters.append(p + (lr / (1 - self.beta1 ** epoch)) * (real_part + 1j * imag_part))
            updated_mt.append(m)
            updated_vt.append(v)
        
        return tuple(updated_parameters + [updated_mt, updated_vt, ut])


class CVAdaGrad(Optimizer):
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        """
        Updates the parameters using the complex-valued AdaGrad optimizer.

        Parameters:
        -----------
        Same as the parent class.

        Returns:
        --------
        tuple
            The updated parameters along with the updated moment estimates.
        """
        updated_parameters, updated_mt = [], []
        
        for p, lr, g, m, v in zip(parameters, learning_rate, gradients, mt, vt):
            m = m + (self.xp.real(g)**2 + 1j*self.xp.imag(g)**2)
            
            real_part = self.xp.real(g) / self.xp.sqrt(self.xp.real(m) + self.epsilon)
            imag_part = self.xp.imag(g) / self.xp.sqrt(self.xp.imag(m) + self.epsilon)
            
            updated_parameters.append(p + lr * (real_part + 1j * imag_part))
            updated_mt.append(m)
        
        return tuple(updated_parameters + [updated_mt, vt, ut])


class AdaGrad(Optimizer):
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        """
        Updates the parameters using the AdaGrad optimizer.

        Parameters:
        -----------
        Same as the parent class.

        Returns:
        --------
        tuple
            The updated parameters along with the updated moment estimates.
        """
        updated_parameters, updated_mt = [], []
        
        for p, lr, g, m in zip(parameters, learning_rate, gradients, mt):
            m = m + (self.xp.abs(g) ** 2)
            updated_parameters.append(p + lr * (g / (self.xp.sqrt(m) + self.epsilon)))
            updated_mt.append(m)
        
        return tuple(updated_parameters + [updated_mt, vt, ut])


class RMSprop(Optimizer):
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        """
        Updates the parameters using the RMSprop optimizer.

        Parameters:
        -----------
        Same as the parent class.

        Returns:
        --------
        tuple
            The updated parameters along with the updated moment estimates.
        """
        updated_parameters, updated_mt = [], []
        
        for p, g, lr, m in zip(parameters, gradients, learning_rate, mt):
            m = m * self.beta1 + (1 - self.beta1) * self.xp.abs(g) ** 2
            updated_parameters.append(p + lr * g / (self.xp.sqrt(m) + self.epsilon))
            updated_mt.append(m)
        
        return tuple(updated_parameters + [updated_mt, vt, ut])


class CVRMSprop(Optimizer):
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        """
        Updates the parameters using the complex-valued RMSprop optimizer.

        Parameters:
        -----------
        Same as the parent class.

        Returns:
        --------
        tuple
            The updated parameters along with the updated moment estimates.
        """
        updated_parameters, updated_mt = [], []
        
        for p, g, lr, m in zip(parameters, gradients, learning_rate, mt):
            m = m * self.beta1 + (1 - self.beta1) * (self.xp.real(g) ** 2 + 1j * self.xp.imag(g) ** 2)
            real_part = self.xp.real(g) / (self.xp.sqrt(self.xp.real(m) + self.epsilon))
            imag_part = self.xp.imag(g) / (self.xp.sqrt(self.xp.imag(m) + self.epsilon))
            updated_parameters.append(p + lr * (real_part + 1j * imag_part))
            updated_mt.append(m)
        
        return tuple(updated_parameters + [updated_mt, vt, ut])


class Nadam(Optimizer):
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        """
        Updates the parameters using the Nadam optimizer.

        Parameters:
        -----------
        Same as the parent class.

        Returns:
        --------
        tuple
            The updated parameters along with the updated moment estimates.
        """
        updated_parameters, updated_mt, updated_vt = [], [], []
        
        for p, g, lr, m, v in zip(parameters, gradients, learning_rate, mt, vt):
            m = m * self.beta1 + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * self.xp.abs(g) ** 2
            
            mt_hat = (1 - self.beta1) * g / (1 - self.beta1 ** (epoch + 1)) + self.beta1 * m / (1 - self.beta1 ** epoch)
            
            updated_parameters.append(p + lr * mt_hat / (self.xp.sqrt(v / (1 - self.beta2 ** epoch)) + self.epsilon))
            updated_mt.append(m)
            updated_vt.append(v)
        
        return tuple(updated_parameters + [updated_mt, updated_vt, ut])


class CVNadam(Nadam):
    def update_parameters(self, parameters, gradients, learning_rate, epoch, mt, vt, ut):
        """
        Updates the parameters using the complex-valued Nadam optimizer.

        Parameters:
        -----------
        parameters : list of arrays
            The parameters of the neural network.
        gradients : list of arrays
            The gradients of the loss function with respect to the parameters.
        learning_rate : float
            The learning rate for updating the parameters.
        epoch : int
            The current epoch number.
        mt : list of arrays
            The first moment estimates.
        vt : list of arrays
            The second moment estimates.
        ut : list of arrays
            The third moment estimates.

        Returns:
        --------
        tuple
            The updated parameters along with the updated moment estimates.
        """
        updated_parameters, updated_mt, updated_vt = [], [], []
        
        for param, grad, lr, m, v in zip(parameters, gradients, learning_rate, mt, vt):
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (self.xp.real(grad) ** 2 + 1j * self.xp.imag(grad) ** 2)
            
            mt_hat = (1 - self.beta1) * grad / (1 - self.beta1 ** (epoch + 1)) + self.beta1 * m / (1 - self.beta1 ** epoch)
            vc = v / (1 - self.beta2 ** epoch)
            
            real_update = self.xp.real(mt_hat) / (self.xp.sqrt(self.xp.real(vc)) + self.epsilon)
            imag_update = self.xp.imag(mt_hat) / (self.xp.sqrt(self.xp.imag(vc)) + self.epsilon)
            
            updated_parameters.append(param + lr * (real_update + 1j * imag_update))
            updated_mt.append(m)
            updated_vt.append(v)
        
        return tuple(updated_parameters + [updated_mt, updated_vt, ut])

