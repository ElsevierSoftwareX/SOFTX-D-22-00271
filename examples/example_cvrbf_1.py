# -*- coding: utf-8 -*-
"""
**RosenPy: An Open Source Python Framework for Complex-Valued Neural Networks**.
*Copyright © A. A. Cruz, K. S. Mayer, D. S. Arantes*.

*License*

This file is part of RosenPy.
RosenPy is an open-source framework distributed under the terms of the GNU General 
Public License, as published by the Free Software Foundation, either version 3 of 
the License, or (at your option) any later version. For additional information on 
license terms, please open the Readme.md file.

RosenPy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with RosenPy.  
If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)


import numpy as np
import rosenpy.model.cvrbfnn as mynn
import rosenpy.utils.utils as utils
import rosenpy.utils.init_func as init_func
import rosenpy.utils.decay_func as decay_func
import rosenpy.model.rp_optimizer as opt
import rosenpy.dataset.beamforming as dt


def set_data():
    """
    Set up the data for training.

    Returns:
        tuple: Tuple containing the normalized input and output datasets.
    """
    f = 850e6
    sinr_db = 20
    snr_dbs = 25
    snr_dbi = 20
    phi = [1, 60, 90, 120, 160, 200, 240, 260, 280, 300, 330]
    theta = [90] * 11
    desired = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    modulation = ["QAM", "WGN", "QAM", "PSK", "QAM", "WGN", "QAM", "WGN", "QAM", "PSK", "PSK"]
    m_mod = [4, 0, 64, 8, 256, 0, 16, 0, 64, 16, 8]
    
    len_data = int(1e4)
    desired = np.array(desired)
    
    set_in, set_out = dt.create_dataset_beam(modulation, m_mod, f, phi, theta, desired, len_data, sinr_db, snr_dbs, snr_dbi)
    
    return set_in, set_out


###############################################################################################################
###############################################################################################################


# Initialize input_data and output_data using the set_data function
input_data, output_data = set_data()

# Create an instance of the CVRBF Neural Network
nn = mynn.CVRBFNN(gpu_enable=False)

# Add layers to the neural network
nn.add_layer(ishape=input_data.shape[1], neurons=20, oshape=20)
nn.add_layer(neurons=20, oshape=output_data.shape[1])

# Train the neural network using fit method
nn.fit(input_data, output_data, epochs=500, verbose=100, batch_size=100, optimizer=opt.CVAdamax())

# Make predictions using the trained model
y_pred = nn.predict(input_data)

# Calculate and print accuracy
print(f'Accuracy: {nn.accuracy(output_data, y_pred):.2f}%')
