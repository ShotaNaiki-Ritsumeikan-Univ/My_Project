import sys
sys.path.append('/usr/local/lib/python3.9/site-packages')

import math
import green_function as gr
import function as fu
import numpy as np
import scipy as sp
from scipy.special import sph_harm
from scipy.special import spherical_jn
from scipy.special import lpmv
from scipy.special import eval_legendre
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import mpl_toolkits.mplot3d.axes3d as axes3d

def d_jn(n, z):
    return spherical_jn(n,z,derivative=True)

def d_hn(n, k, z):
    return -fu.shankel(n+1, k, z)+(n/z)*fu.shankel(n, k, z)

def calc_mode_strength(N,k,a,r_sp):
    kr = k*r_sp
    ka = k*a
    if isinstance(k, float) == True:    #単一周波数の場合
        mode_strength = np.array([],dtype = np.complex128)
        for n in range(N+1):
            for m in range(-n, n+1):
                mode_strength = np.append(mode_strength,spherical_jn(n,kr)/(ka*a*d_jn(n,ka)))
    elif isinstance(k, float) == False:   #複数周波数の場合
        mode_strength = np.array([],dtype = np.complex128)
        for n in range(N+1):
            for m in range(-n, n+1):
                mode_strength = np.append(mode_strength,spherical_jn(n,kr)/(ka*a*d_jn(n,ka)))
        mode_strength = mode_strength.reshape((N+1)**2,len(k)).T
    return mode_strength

def calc_mode_strength_out(N,k,a,r_sp):
    kr = k*r_sp
    ka = k*a
    if isinstance(k, float) == True:    #単一周波数の場合
        mode_strength = np.array([],dtype = np.complex128)
        for n in range(N+1):
            for m in range(-n, n+1):
                mode_strength = np.append(mode_strength,4*np.pi*fu.shankel(n,0,kr)/(ka*a*d_hn(n,0,ka)))
    elif isinstance(k, float) == False:   #複数周波数の場合
        mode_strength = np.array([],dtype = np.complex128)
        for n in range(N+1):
            for m in range(-n, n+1):
                mode_strength = np.append(mode_strength,4*np.pi*fu.shankel(n,0,kr)/(ka*a*d_hn(n,0,ka)))
        mode_strength = mode_strength.reshape((N+1)**2,len(k)).T
    return mode_strength

def sph_harm_m0(n, theta):
    cos = np.cos(theta)
    Pn = eval_legendre(n, cos)
    _sph_harm_m0 = np.sqrt((2*n+1)/4*np.pi)*Pn
    return _sph_harm_m0

def calcAnm(N, desired_phi, desired_theta):
    d_phi = np.deg2rad(desired_phi)
    d_theta = np.pi/2 - np.deg2rad(desired_theta)

    c1 = np.array([],dtype = np.complex128)
    c2 = np.array([],dtype = np.complex128)
    # c3 = np.array([],dtype = np.complex128)
    # c4 = np.array([],dtype = np.complex128)
    # for n in range(N+1):
    #     for m in range(-n, n+1):
            # if m == 0:
            #     c1 = np.append(c1,sph_harm_m0(n, d_theta))
            #     c2 = np.append(c2,sph_harm_m0(n, np.pi-d_theta))
            # elif    m < 0:
            #     c1 = np.append(c1,1j*np.imag(sph_harm(m, n, d_phi, d_theta)))
            #     c2 = np.append(c2,1j*np.imag(sph_harm(m, n, d_phi+(np.pi/2), np.pi-d_theta)))
            # elif    m > 0:
            #     c1 = np.append(c1,np.real(sph_harm(m, n, d_phi, d_theta)))
            #     c2 = np.append(c2,np.real(sph_harm(m, n, d_phi+(np.pi/2), np.pi-d_theta)))
    c1 = calcSH(N, d_phi, d_theta)
    if d_phi<=np.pi:
        c2 = calcSH(N, d_phi+np.pi, np.pi-d_theta)
    else:
        c2 = calcSH(N, d_phi-np.pi, np.pi-d_theta)
    # c3 = calcSH(N, d_phi+6/np.pi, d_theta)
    # c4 = calcSH(N, d_phi-6/np.pi, d_theta)
    C = np.append(c1, c2).reshape(2,(N+1)**2).T
    # C = np.append(C, c3).reshape(3,(N+1)**2).T
    # C = np.append(C, c4).reshape(4,(N+1)**2).T
    f = np.array([1,0]).T

    t_num = 90
    p_num = 180
    theta, phi = np.mgrid[0:np.pi:1j*t_num, 0:2*np.pi:1j*p_num]

    y_nm = calcSH(N, phi, theta)
    R = y_nm@np.conj(y_nm).T

    Anm = np.linalg.pinv(R)@C@np.linalg.pinv((np.conj(C).T@np.linalg.pinv(R)@C))@f
    return Anm

def calcSH(N, azi, ele):
    y_n = np.array([],dtype = np.complex128)
    if azi.ndim == 0:
        for n in range(N+1):
            for m in np.arange(-n,n+1,1):
                y_n = np.append(y_n,sph_harm(m,n,azi,ele))
        return y_n.T
    elif len(azi)<2080:
        for n in range(N+1):
            for m in np.arange(-n,n+1,1):
                y_n = np.append(y_n,sph_harm(m,n,azi,ele))
        return y_n.reshape((N+1)**2,len(azi))
    else:
        for n in range(N+1):
            for m in np.arange(-n,n+1,1):
                y_n = np.append(y_n,sph_harm(m,n,azi,ele))
        return y_n.reshape((N+1)**2,len(azi)*len(ele)*2)

def calcDirWeights(num_mic, r_source, mic_xyz): #Cardioid Weights
    B = np.zeros(num_mic).reshape(num_mic,)

    for i in range(num_mic):
        a = np.linalg.norm(r_source)*np.linalg.norm(mic_xyz[i,:])
        b = np.inner(r_source, mic_xyz[i,:])
        x = b/a

        rad = np.arccos(x)
        B[i] = rad

    # weight = (1 + np.cos(B))/2
    alpha = 0.5
    weight = alpha + (1-alpha)*np.cos(B)
    return weight

def calcDirWeights2(sp_xyz, mic_xyz): #Cardioid Weights
    a = np.linalg.norm(sp_xyz)*np.linalg.norm(mic_xyz)
    b = np.inner(sp_xyz, mic_xyz)

    alpha = 0.5
    # rad = np.arccos(b/a)
    # weight = alpha + (1-alpha)*np.cos(rad)
    weight = alpha + (1-alpha)*(b/a)
    return weight
