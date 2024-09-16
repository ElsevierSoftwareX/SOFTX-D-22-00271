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

import numpy as np

def create_dataset_beam(modulation, Mmod, f, phi, theta, desired, lenData, SINRdB, SNRdBs, SNRdBi):
    """
    Create a dataset for beamforming.

    Args:
        modulation (list): List of modulation types.
        Mmod (list): List of modulation orders.
        f (float): Frequency of the signal.
        phi (float): Angle in degrees for the transmitter.
        theta (float): Angle in degrees for the elevation.
        desired (array): Array indicating if each output is desired or not.
        lenData (int): Length of the dataset.
        SINRdB (float): Signal-to-Interference-plus-Noise Ratio in dB.
        SNRdBs (float): Signal-to-Noise Ratio in dB for sources.
        SNRdBi (float): Signal-to-Noise Ratio in dB for interferences.

    Returns:
        tuple: Tuple containing the input and output datasets.
    """
    # Linear SNR of sources
    SNRs = 10**(SNRdBs/10)
    # Linear SNR of interferences
    SNRi = 10**(SNRdBi/10)
    # Linear SINR
    SINR = 10**(SINRdB/10)

    # Number of sources
    Ns = np.sum(desired)
    # Number of interferences
    Ni = len(desired) - Ns

    # Normalization factor to the desired SINR
    iota = (Ni/Ns) * (1/SNRi + 1) / (1/SINR - 1/SNRs)

    # Standard deviations for AWGN noises of sources and interferences
    StdDevS = np.sqrt((1/SNRs)/2)
    StdDevI = np.sqrt((1/(iota*SNRi))/2)

    # Create StdDevVect for each modulation type
    #StdDevVect = np.zeros((len(modulation), lenData), dtype=complex)

    StdDevVect = np.array([np.where(desired == 1, StdDevS, StdDevI)]).T


    # Speed of light in vacuum [m/s]
    c = 299792458
    # Wavelength
    lambda_val = c / f
    # Propagation constant
    beta = 2 * np.pi / lambda_val
    # Dipoles length
    L = 0.5 * lambda_val
    # Dipoles spacing
    s = 0.25 * lambda_val
    # Dipoles coordinates
    coord = np.array([
        [s, 0, 0],
        [s * np.cos(np.deg2rad(60)), s * np.sin(np.deg2rad(60)), 0],
        [-s * np.cos(np.deg2rad(60)), s * np.sin(np.deg2rad(60)), 0],
        [-s, 0, 0],
        [-s * np.cos(np.deg2rad(60)), -s * np.sin(np.deg2rad(60)), 0],
        [s * np.cos(np.deg2rad(60)), -s * np.sin(np.deg2rad(60)), 0]
    ]).T

    # Matrix of self and mutual impedances
    Z = np.array([
        [78.424+45.545j, 46.9401-32.6392j, -0.791-41.3825j, -14.4422-34.4374j, -0.791-41.3825j, 46.9401-32.6392j],
        [46.9401-32.6392j, 78.424+45.545j, 46.9401-32.6392j, -0.791-41.3825j, -14.4422-34.4374j, -0.791-41.3825j],
        [-0.791-41.3825j, 46.9401-32.6392j, 78.424+45.545j, 46.9401-32.6392j, -0.791-41.3825j, -14.4422-34.4374j],
        [-14.4422-34.4374j, -0.791-41.3825j, 46.9401-32.6392j, 78.424+45.545j, 46.9401-32.6392j, -0.791-41.3825j],
        [-0.791-41.3825j, -14.4422-34.4374j, -0.791-41.3825j, 46.9401-32.6392j, 78.424+45.545j, 46.9401-32.6392j],
        [46.9401-32.6392j, -0.791-41.3825j, -14.4422-34.4374j, -0.791-41.3825j, 46.9401-32.6392j, 78.424+45.545j]
    ])

    # Load impedance
    ZT = 50  # [ohms]

    # Dipoles self impedance
    ZA = Z[0, 0]

    # Coupling matrix
    C = (ZT + ZA) * np.linalg.inv((Z + ZT * np.eye(6)))

    # Matrix of relative intensity of Etheta
    Xm = np.diag((lambda_val / (np.pi * np.sin(beta * L / 2))) * ((np.cos(L * np.pi * np.cos(np.deg2rad(theta)) / lambda_val) - np.cos(np.pi * L / lambda_val)) / np.sin(np.deg2rad(theta))))

    # Matrix of Tx angular positions
    Omega = np.array([np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(theta))]).T

    Pi = np.exp(-2j * np.pi * np.dot(Omega, coord) / lambda_val)

    # Steering vectors
    psi = np.dot(Pi.T, Xm)

    # Create symbols of sources and interferences
    SetOut = np.zeros((len(modulation), lenData), dtype=complex)

    for ii, mod_type in enumerate(modulation):
        # QAM symbols
        if mod_type == "QAM":
            if Mmod[ii] != 0:
                SetOut[ii] = (((np.random.randint(np.sqrt(Mmod[ii]), size=lenData)) * 2) - (np.sqrt(Mmod[ii]) - 1)) + \
                            1j * (((np.random.randint(np.sqrt(Mmod[ii]), size=lenData) ) * 2) - (np.sqrt(Mmod[ii]) - 1))
            else:
                SetOut[ii] = np.random.randn(1, lenData) + 1j * np.random.randn(1, lenData)
        # PSK symbols
        elif mod_type == "PSK":
            if Mmod[ii] != 0:
                pskAng = np.random.randint(Mmod[ii], size=lenData) * 2 * np.pi / Mmod[ii]
                SetOut[ii] = np.cos(pskAng) + 1j * np.sin(pskAng)
            else:
                SetOut[ii] = np.random.randn(1, lenData) + 1j * np.random.randn(1, lenData)
        # WGN noise
        else:
            SetOut[ii] = np.random.randn(1, lenData) + 1j * np.random.randn(1, lenData)

    # Compute the source and interference powers for normalization
    P = np.sum((np.abs(SetOut - np.mean(SetOut, axis=1)[:, np.newaxis])**2), axis=1) / lenData

    # Normalize the powers to 1
    SetOut = SetOut / np.sqrt(P[:, np.newaxis])

    # Interference normalizations to the desired SINR
    #SetOut[~desired] = SetOut[~desired] / np.sqrt(iota)
    SetOut[np.where(desired == 0)[0],:] = SetOut[np.where(desired == 0)[0],:] / np.sqrt(iota)
    #SetOut[~desired] = SetOut[~desired] / np.sqrt(iota)
    
    # Create the data that impinged on the beamforming
    SetIn = np.dot(C, psi).dot(SetOut + StdDevVect * (np.random.randn(len(modulation), lenData) + 1j * np.random.randn(len(modulation), lenData)))

    # Compute the signal powers in each RX dipole
    P = np.sum((np.abs(SetIn - np.mean(SetIn, axis=1)[:, np.newaxis])**2), axis=1) / lenData

    # Normalize the signals in each dipole to have a unitary power
    SetIn = SetIn / np.sqrt(P[:, np.newaxis])

    # Add nonlinearities
    SetIn = SetIn - 0.1 * SetIn**3 - 0.05 * SetIn**5

    # Select desired outputs
    #SetOut = SetOut[desired]
    indices_true = np.where(desired)
    SetOut = SetOut[indices_true]

    return SetIn.T, SetOut.T
