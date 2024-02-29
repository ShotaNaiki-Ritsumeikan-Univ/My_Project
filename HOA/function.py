import sys
sys.path.append('/usr/local/lib/python3.9/site-packages')

import scipy as sp
from scipy.special import sph_harm
import numpy as np
from scipy.special import spherical_jn
from scipy.special import spherical_yn
from scipy.special import eval_legendre
from numpy import arctan2, sqrt
import numexpr as ne
import sympy
from sympy.physics.wigner import gaunt

def cart2sph(x,y,z):
    azimuth = ne.evaluate('arctan2(y,x)')
    xy2 = ne.evaluate('x**2 + y**2')
    elevation = ne.evaluate('arctan2(z, sqrt(xy2))')
    r = eval('sqrt(xy2 + z**2)')
    return azimuth, elevation, r

def EquiDistSphere(n):
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * (np.arange(n)+1)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:,0] = radius * np.cos(theta)
    points[:,1] = radius * np.sin(theta)
    points[:,2] = z
    return points

def BnMat2(N,k,r):
    B = np.zeros((N+1)**2, dtype = np.complex128).reshape((N+1)**2,)
    #(N+1)^2個の要素を持つ複素配列を定義
    dx = 0
    for i in range(N+1):    #0~(N+1)-1まで
        A = Bn2(i,k,r)
        for j in np.arange(-i,i+1,1):
            B[dx] = A
            dx +=1
    return B

def Bn2(n,k,r):
    kr = k*r
    y = 4 * np.pi * (1j**(n)) * spherical_jn(n,kr)
    return y

def shankel(n,k,z):
    if k == 1:
        sha = spherical_jn(n,z) + 1j*spherical_yn(n,z)
    else:
        sha = spherical_jn(n,z) - 1j*spherical_yn(n,z)
    return sha

def calcSH(N,azi,ele):
    y_n = np.array([],dtype = np.complex128)
    for n in range(N+1):
        for m in np.arange(-n,n+1,1):
            y_n = np.append(y_n,sph_harm(m,n,azi,ele))
    return y_n

def encode(ambi_order, azi, ele, k, r, mic_sig, enc):
    Y_N = np.zeros((len(azi),(ambi_order+1)**2),dtype = np.complex128)
    for i in range(azi.size):
        Y_N[i,:] = calcSH(ambi_order, azi[i],ele[i])
    invY = np.linalg.pinv(Y_N)  #球面調和関数の一般化逆行列

    if enc == 0:   #Open sphere エンコード
        Wnkr = BnMat2(ambi_order,k,r)
    elif enc == 1: #Cardioid エンコード
        Wnkr = BnMat2cardioid(ambi_order,k,r)

    ambi_signal = np.diag(1/Wnkr) @ invY @ mic_sig
    return ambi_signal

def decode(ambi_order, azi, ele, k, r, lambda_, ambi_signal):
    Y_N = np.zeros((len(azi),(ambi_order+1)**2),dtype = np.complex128)
    for i in range(azi.size):
        Y_N[i,:] = calcSH(ambi_order, azi[i],ele[i])
    invY = np.linalg.pinv(np.conj(Y_N.T))

    Fnkr = calcFnkr(ambi_order,k,r,r)   #球面波デコード
    spkr_output = invY @ np.diag(1/Fnkr) @ ambi_signal

    return spkr_output

def d_jn(n, z):
    return spherical_jn(n,z,derivative=True)

def d_hn(n, k, z):
    return -shankel(n+1, k, z)+(n/z)*shankel(n, k, z)

def d_hn(n, k, z):
    return -shankel(n+1, k, z)+(n/z)*shankel(n, k, z)

def calcFnkr(N,k,r,a):
    B = np.zeros((N+1)**2, dtype = np.complex128).reshape((N+1)**2,)

    dx = 0
    for i in range(N+1):    #0~(N+1)-1まで
        A = shankel(i,0,k*r)
        for j in np.arange(-i,i+1,1):
            B[dx] = 1j*k*A/(1j**i)
            # B[dx] = A
            dx +=1
    return B

def BnMat2cardioid(N,k,r):
    B = np.zeros((N+1)**2, dtype = np.complex128).reshape((N+1)**2,)

    dx = 0
    for i in range(N+1):    #0~(N+1)-1まで
        A = 0.5*spherical_jn(i,k*r)-0.5*1j*d_jn(i,k*r)   #球面波エンコード・デコード用
        for j in np.arange(-i,i+1,1):
            B[dx] = 4*np.pi*(1j**i)*A
            # B[dx] = A
            dx +=1
    return B

def calcHankel(N,k,r):
    B = np.zeros((N+1)**2, dtype = np.complex128).reshape((N+1)**2,)

    dx = 0
    for i in range(N+1):    #0~(N+1)-1まで
        A = shankel(i,0,k*r)
        for j in np.arange(-i,i+1,1):
            B[dx] = A
            dx +=1
    return B

def calcPext(N, k, x, y, z):
    azi, ele, r = cart2sph(x, y, z)
    # azi = azi + rotate_angle  #rotate_angleで放射方向を回転できる
    ele = np.pi/2 - ele
    return np.sqrt(4*np.pi)*calcHankel(N,k,r)*calcSH(N,azi,ele)
