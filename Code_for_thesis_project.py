import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import eig
from scipy.linalg import expm
from scipy.sparse import diags
from numpy.linalg import norm
from scipy.linalg import eig
from scipy.linalg import block_diag
from numpy import cast
import random
from numba import jit


size = 501 #Number of sites total
probe_num = 0 
init_fillin = 101 
phi_real = 10
hopping = 1.0 #J
gamma = 1.0 #strength of probe
lbd = 0.5 # quasi_periodic_parameter
irrb = (1 + np.sqrt(5))/2
siteindx = np.array(range(1, size+1))
sitepotential = 2*lbd*np.cos(2*np.pi*irrb*(siteindx))
diagonals = [sitepotential,-1.0*hopping*np.ones(size-1), -1.0*hopping*np.ones(size-1)]
offset = [0,-1,1]
sys_Ham = diags(diagonals,offset,dtype='float_').toarray()
time = [0.0, 25.0, 50.0, 100.0]
phi_val = np.linspace(0 , 2*np.pi, phi_real)


def relabelling(array): #reshuufling of the eigenectors
    size = int(len(array))
    arraynew = np.zeros(size, dtype = complex)
    cent = int((((size + 1)/2) - 1))
    arraynew[cent] = array[0]
    A = []
    for i in range(1, size - 1, 2):
        A.append(i)
    for i in A :
        arraynew[(i + (len(A) - A.index(i)))] = array[i]
    B = []
    for i in range(2, size, 2):
        B.append(i)
    for i in B:
        arraynew[((cent - B.index(i)) - 1)] = array[i]
    return (arraynew)


def init_lyapunov_centdense():
    initdens = np.zeros((size,size), dtype = complex)
    for i in range(0,init_fillin):
        initdens[i][i] = 1.0 + 0.0*1j
    daig = np.diag(initdens)
    newdiag = relabelling(daig)
    initdens = np.diag(newdiag,k = 0) 
    fullarray = np.zeros((phi_real, size, size), dtype = complex)
    for i in range(phi_real):
        for j in range(size):
            for k in range(size):
                fullarray[i,j,k] = initdens[j,k]
    print('done1')
    return fullarray

D_matrix = np.zeros((size,size), dtype = complex)
for i in range(0,probe_num):
    D_matrix[i][i] = 1.0 + 0.0*1j
daig = np.diag(D_matrix)
newdiag = relabelling(daig)
D_matrix =  np.diag(newdiag,k = 0) 
D_matrix = D_matrix*(gamma/2)
D_matrix = D_matrix



guy_mat = np.zeros((size,size), dtype = complex)
for i in range(0,probe_num):
    guy_mat[i][i] = 1.0 + 0.0*1j
daig = np.diag(guy_mat)
newdiag = relabelling(daig)
guy_mat =  np.diag(newdiag,k = 0)
guy_mat = guy_mat


#@jit(nopython=True)
def runga_kutta_step(itime, ival, ftime):
    dt = 10**(-2)
    initdens = ival
    for j in range(phi_real):
        hopping = 1.0 #J
        siteindx = np.array(range(1, size+1))
        sitepotential = 2*lbd*np.cos(2*np.pi*irrb*(siteindx) + phi_val[j])
        site_potent_diag = np.diag(sitepotential, k=0)
        hopping_right = np.diag(-1.0*hopping*np.ones(size-1), k=1)
        hopping_left = np.diag(-1.0*hopping*np.ones(size-1), k=-1)
        sys_Ham = site_potent_diag +  hopping_right + hopping_left
        W_matrix = sys_Ham - 1j * D_matrix
        t = itime
        correlate = initdens[j,:,:]
        print('done2')
        while t < ftime:
            step_one   = dt*(-1j*(W_matrix@correlate) + 1j*(correlate@np.conjugate(W_matrix.T)) + (gamma*(guy_mat*(correlate))))
            step_two   = dt*(-1j*(W_matrix@(correlate + (step_one/2) )) + 1j*((correlate + (step_one/2))@np.conjugate(W_matrix.T)) + (gamma*(guy_mat*(correlate + step_one/2))))
            step_three = dt*(-1j*(W_matrix@(correlate + (step_two/2) )) + 1j*((correlate + (step_two/2))@np.conjugate(W_matrix.T)) + (gamma*(guy_mat*(correlate + step_two/2))))
            step_four  = dt*(-1j*(W_matrix@(correlate +   step_three )) + 1j*((correlate + (step_three))@np.conjugate(W_matrix.T)) + (gamma*(guy_mat*(correlate +step_three))))
            correlate  = correlate + (step_one + 2*step_two + 2*step_three + step_four)/6
            t += dt
        initdens[j,:,:] = correlate
        print('done3')
    return initdens

    
realname = "{i}.delocalise_correlation_no_probe__real.txt"
imagname = "{i}.delocalise_correlation_no_probe__imag.txt"
ltdens = init_lyapunov_centdense()
save_init = np.average(ltdens, axis = 0)
np.savetxt(realname.format(i=0), save_init.real)
np.savetxt(imagname.format(i=0), save_init.imag)
for i in range(1, len(time)):
    dens = runga_kutta_step(time[i-1], ltdens, time[i])
    save_dens = np.sum(dens, axis = 0)/(phi_real)
    np.savetxt(realname.format(i=i), save_dens.real)
    np.savetxt(imagname.format(i=i), save_dens.imag)
    ltdens = dens
    print('done_{i}')