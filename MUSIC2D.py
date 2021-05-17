#!/usr/bin/env python3

import numpy as np

def MUSIC2D(data, source_num, width, sig, HR_grid_size):

    # Recreate low-resolution sensors (assuming ULA)
    sensor_x = np.linspace(-width,width, int(np.sqrt(data.shape[0]))).reshape(-1,1)
    sensor_y = np.linspace(-width,width, int(np.sqrt(data.shape[0]))).reshape(-1,1)
    sensor_pos = np.stack(np.meshgrid(sensor_x,sensor_y),-1).reshape(-1,2)

    # Perform SVD on the data
    U,s,_ = np.linalg.svd(data, full_matrices=True)
  
    # Select columns of U which correspond to noise-subspace
    U2 = U[:,source_num:]
  
    # Create Grid for MUSIC
    HR_sensor_x = np.linspace(-width,width,HR_grid_size).reshape(-1,1)
    HR_sensor_y = np.linspace(-width,width,HR_grid_size).reshape(-1,1)
    HR_sensor_pos = np.stack(np.meshgrid(HR_sensor_x,HR_sensor_y),-1).reshape(-1,2)

    # Create HR sensing matrix
    arg = np.abs(sensor_pos[:,np.newaxis] - HR_sensor_pos)
    A = np.exp(-(0.5/sig**2)*np.linalg.norm(arg,axis=2)**2)

    # Calculate error between sensor image and projection, and invert for MUSIC result
    result = U2.conj().T @ A
    err = np.linalg.norm(result, axis=0)**2
    music = 1/err
  
    return music
