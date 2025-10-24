'''
Utility functions for UR5e MuJoCo simulation
Based on Simple-MuJoCo-PickNPlace utilities
'''

import time
import math
import numpy as np

def rot_mtx(deg):
    """
        2 x 2 rotation matrix
    """
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def pr2t(p,R):
    """ 
        Convert pose to transformation matrix 
    """
    p0 = p.ravel() # flatten
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def t2pr(T):
    """
        T to p and R
    """   
    p = T[:3,3]
    R = T[:3,:3]
    return p,R

def t2p(T):
    """
        T to p 
    """   
    p = T[:3,3]
    return p

def t2r(T):
    """
        T to R
    """   
    R = T[:3,:3]
    return R

def rpy2r(rpy_rad):
    """
        Roll-pitch-yaw to rotation matrix
    """
    roll  = rpy_rad[0]
    pitch = rpy_rad[1] 
    yaw   = rpy_rad[2]
    Cphi  = np.cos(roll)
    Sphi  = np.sin(roll)
    Cthe  = np.cos(pitch)
    Sthe  = np.sin(pitch)
    Cpsi  = np.cos(yaw)
    Spsi  = np.sin(yaw)
    R = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    return R

def r2rpy(R,unit='rad'):
    """
        Rotation matrix to roll-pitch-yaw
    """
    roll  = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], (math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    yaw   = math.atan2(R[1, 0], R[0, 0])
    if unit == 'deg':
        roll  = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        yaw   = np.rad2deg(yaw)
    return np.array([roll, pitch, yaw])

def r2w(R):
    """
        Rotation matrix to angular velocity (axis-angle)
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.math.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()

def trim_scale(x,th):
    """
        Trim scale - limit maximum absolute value
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x

def deg2rad(deg):
    """
        Degree to radian
    """
    return deg*np.pi/180.0

def rad2deg(rad):
    """
        Radian to degree  
    """
    return rad*180.0/np.pi

def soft_squash_multidim(x, x_min, x_max, margin=0.1):
    """
        Multi-dimensional soft squashing function
    """
    x_in = np.copy(x)
    x_out = np.copy(x)
    for i in range(len(x)):
        x_out[i] = soft_squash(x_in[i], x_min[i], x_max[i], margin)
    return x_out    

def soft_squash(x, x_min, x_max, margin=0.1):
    """
        Soft squashing function
    """
    def th(z,m=0.0):
        return np.tanh((z-m)/2.0)
    x_in = np.copy(x)
    x_range = x_max - x_min
    x_mid = (x_max + x_min) / 2.0
    x_scaled = (x_in - x_mid) / x_range
    x_squash = th(x_scaled,m=margin) * (x_range/2.0) * (1.0-margin) + x_mid
    return x_squash
