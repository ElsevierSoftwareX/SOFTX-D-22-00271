# -*- coding: utf-8 -*-
"""
RosenPy: An Open Source Python Framework for Complex-Valued Neural Networks.
Copyright © A. A. Cruz, K. S. Mayer, D. S. Arantes.

License:
This file is part of RosenPy.
RosenPy is an open source framework distributed under the terms of the GNU 
General Public License, as published by the Free Software Foundation, either 
version 3 of the License, or (at your option) any later version. For additional 
information on license terms, please open the Readme.md file.

RosenPy is distributed in the hope that it will be useful to every user, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
details.

You should have received a copy of the GNU General Public License along with 
RosenPy. If not, see <http://www.gnu.org/licenses/>.
"""

from rosenpy.utils import cost_func, decay_func, batch_gen_func, select_module
from .rp_layer import Layer
from . import rp_optimizer as opt

class NeuralNetwork:
    """
    Abstract base class for wrapping all neural network functionality from 
    RosenPy. This is a superclass.
    """
   
    def __init__(self, cost_func=cost_func.mse, patience=float('inf'), 
                 gpu_enable=False):
        """
        Initializes the neural network with default parameters.

        Parameters:
        -----------
        cost_func : function, optional
            The cost function to be used for training the neural network. 
            Default is mean squared error (MSE).
        patience : int, optional
            The patience parameter for early stopping during training. 
            Default is a large value to avoid early stopping.
        gpu_enable : bool, optional
            Flag indicating whether GPU acceleration is enabled. Default is 
            False.
        """
        self.xp = select_module(gpu_enable)
        self.gpu_enable = gpu_enable
        self.layers = []
        self.cost_func = cost_func

        self.optimizer = None
        self.patience, self.waiting = patience, 0

        self._best_model, self._best_loss = self.layers, self.xp.inf
        self._history = {'epochs': [], 'loss': [], 'loss_val': []}

    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=100, 
            verbose=10, batch_gen=batch_gen_func.batch_sequential, 
            batch_size=1, optimizer=opt.GradientDescent(beta=100, beta1=0.9, 
                                                        beta2=0.999)):
        """
        Trains the neural network on the provided training data.

        Parameters:
        -----------
        x_train : array-like
            The input training data.
        y_train : array-like
            The target training data.
        x_val : array-like, optional
            The input validation data. Default is None.
        y_val : array-like, optional
            The target validation data. Default is None.
        epochs : int, optional
            The number of training epochs. Default is 100.
        verbose : int, optional
            Controls the verbosity of the training process. Default is 10.
        batch_gen : function, optional
            The batch generation function to use during training. Default is 
            batch_gen_func.batch_sequential.
        batch_size : int, optional
            The batch size to use during training. Default is 1.
        optimizer : Optimizer, optional
            The optimizer to use during training. Default is GradientDescent 
            with specified parameters.
        """
        self.verify_input(x_train)
        self.optimizer = optimizer
        self.optimizer.set_module(self.xp)

        x_train, y_train = self.convert_data(x_train), self.convert_data(y_train)

        self.mean_in = self.xp.mean(x_train)
        self.mean_out = self.xp.mean(y_train)

        self.std_in = self.xp.std(x_train)
        self.std_out = self.xp.std(y_train)

        x_val, y_val = (x_train, y_train) if (x_val is None or y_val is None) else (
            self.convert_data(x_val), self.convert_data(y_val))

        x_train = self.normalize_data(x_train, self.mean_in, self.std_in)
        y_train = self.normalize_data(y_train, self.mean_out, self.std_out)
        x_val = self.normalize_data(x_val, self.mean_in, self.std_in)
        y_val = self.normalize_data(y_val, self.mean_out, self.std_out)

        for epoch in range(1, epochs + 1):
            x_batch, y_batch = batch_gen(self.xp, x_train, y_train, batch_size)
            self.update_learning_rate(epoch)

            for x_batch1, y_batch1 in zip(x_batch, y_batch):
                y_pred = self.feedforward(x_batch1)
                self.backprop(y_batch1, y_pred, epoch)

            loss_val = self.cost_func(self.xp, y_val, self.predict(x_val, status=0))

            if self.patience != float('inf'):
                if loss_val < self._best_loss:
                    self._best_model, self._best_loss = self.layers, loss_val
                    self.waiting = 0
                else:
                    self.waiting += 1
                    print(f"Not improving: [{self.waiting}] current loss val: "
                          f"{loss_val} best: {self._best_loss}")

                    if self.waiting >= self.patience:
                        self.layers = self._best_model
                        print(f"Early stopping at epoch {epoch}")
                        return

            if epoch % verbose == 0:
                loss_train = self.cost_func(self.xp, y_train, 
                                            self.predict(x_train, status=0))
                self._history['epochs'].append(epoch)
                self._history['loss'].append(loss_train)
                self._history['loss_val'].append(loss_val)
                print(f"Epoch: {epoch:4}/{epochs} loss_train: {loss_train:.8f} "
                      f"loss_val: {loss_val:.8f}")

        return self._history

    def predict(self, x, status=1):
        """
        Predicts the output for the given input data.

        Parameters:
        -----------
        x : array-like
            The input data for prediction.

        Returns:
        --------
        array-like
            The predicted output for the input data.
        """
        if status:
            input_data = self.normalize_data(self.convert_data(x), 
                                             self.mean_in, self.std_in)
            output = self.feedforward(input_data)
            return self.denormalize_outputs(output, self.mean_out, self.std_out)
        else:
            return self.feedforward(self.convert_data(x))

    def accuracy(self, y, y_pred):
        """
        Computes the accuracy of the predictions.

        Parameters:
        -----------
        y : array-like
            The true labels or target values.
        y_pred : array-like
            The predicted values.

        Returns:
        --------
        float
            The accuracy of the predictions as a percentage.
        """
        if isinstance(y, type(y_pred)) and isinstance(y_pred, type(y)):
            return 100 * (1 - self.xp.mean(self.xp.abs(y - y_pred)))
        else:
            print("Datas have different types.")
            return 0

    def add_layer(self):
        pass

    def update_learning_rate(self, epoch):
        """
        Updates the learning rates of all layers based on the current epoch.

        Parameters:
        -----------
        epoch : int
            The current epoch number.
        """
        
        for layer in self.layers:
            for i in range(len(layer.learning_rates)):  
                layer.learning_rates[i] = layer.lr_decay_method(
                    layer.learning_rates[i], epoch, 
                    layer.lr_decay_rate, layer.lr_decay_steps)
                
    def _get_optimizer(self, optimizer_class):
        """
        Creates an instance of the specified optimizer class.

        Parameters:
        -----------
        optimizer_class : class
            The class of the optimizer to be instantiated.

        Returns:
        --------
        instance
            An instance of the specified optimizer class.
        """
        return optimizer_class()

    def verify_input(self, data):
        """
        Verifies the input data type for optimal performance of the RosenPY 
        framework.

        Parameters:
        -----------
        data : array-like
            The input data.
        """
        if not isinstance(data, self.xp.ndarray):
            print("For optimal performance of the RosenPY framework, when not "
                  "using GPU, input the data in NUMPY format, and when utilizing "
                  "GPU, input the data in CUPY format.\n\n")

    def convert_data(self, data):
        """
        Converts the input data to the appropriate format for the current backend 
        (NUMPY or CUPY).

        Parameters:
        -----------
        data : array-like
            The input data.

        Returns:
        --------
        array-like
            The converted input data.
        """
        if isinstance(data, self.xp.ndarray):
            return data
        if self.xp.__name__ == "cupy":
            return self.xp.asarray(data)
        if self.xp.__name__ == "numpy":
            return data.get()
        raise ValueError("Unsupported data type")

    
    def get_history(self):
        """
        Returns the training history of the neural network.
    
        Returns:
        --------
        dict
            A dictionary containing the training history.
        """
        return self._history
    
    def normalize_data(self, input_data, mean=0, std_dev=0):
        """
        Normalizes the input data based on the provided mean and standard deviation.
    
        Parameters:
        -----------
        input_data : array-like
            The data to be normalized.
        mean : float, optional
            The mean for normalization. Default is 0.
        std_dev : float, optional
            The standard deviation for normalization. Default is 0.
    
        Returns:
        --------
        array-like
            The normalized data.
        """
        return input_data
    
    def denormalize_outputs(self, normalized_output_data, mean=0, std_dev=0):
        """
        Denormalizes the output data based on the provided mean and standard deviation.
    
        Parameters:
        -----------
        normalized_output_data : array-like
            The data to be denormalized.
        mean : float, optional
            The mean used for normalization. Default is 0.
        std_dev : float, optional
            The standard deviation used for normalization. Default is 0.
    
        Returns:
        --------
        array-like
            The denormalized data.
        """
        return normalized_output_data
    
        