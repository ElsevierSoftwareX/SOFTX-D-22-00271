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


def split_set(module, x, y, train_size=0.7, random_state=None):
    """
    Split the dataset into training and test sets.

    Parameters
    ----------
    module : module
        The backend module (e.g., NumPy or CuPy).
    x : array-like
        Input data.
    y : array-like
        Target data.
    train_size : float, optional
        Proportion of the dataset to include in the training set (default is 0.7).
    random_state : int, optional
        Seed used by the random number generator (default is None).

    Returns
    -------
    x_train : array-like
        Training data.
    y_train : array-like
        Training target data.
    x_test : array-like
        Test data.
    y_test : array-like
        Test target data.
    """
    if random_state is not None:
        rand_state = module.random.RandomState(random_state)
        rand_state.shuffle(x)
        rand_state.seed(random_state)
        rand_state.shuffle(y)

    split = int(train_size * x.shape[0])
    
    x_train = x[:split]
    y_train = y[:split]
    x_test = x[split:]
    y_test = y[split:]
    
    return x_train, y_train, x_test, y_test
