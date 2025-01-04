# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:15:03 2020

@author: Artem
"""

import numpy as np
from scipy.integrate import quad
from scipy.integrate import fixed_quad
from scipy.optimize import approx_fprime as num_dev

def z(r,phi):
    nominator = 2*np.sqrt(1-(r**2))*r*np.sin(phi+np.pi/4)
    denominator = np.sqrt(1-(r**4)*(np.cos(2*phi)**2))
    return nominator/denominator

def eps_k(k):   #tb dispersion
    return -2*t*np.cos(k*a)

def E_tot_approx(x,kF,mu):  #helper function for numerical derivative
    r,phi,lagr = x
    E_k = lambda k : 1/2*(eps_k(k)+eps_f+lagr
                          - np.sqrt((eps_k(k)-eps_f-lagr)**2 + 4*(V*z(r,phi))**2)) - mu # -mu?
    integrand = lambda k : 1/(2*np.pi) * E_k(k) * np.heaviside(-E_k(k), 0) #0/1 in 2nd arg? -> does not matter (Nullmass)
    integral, err = quad(integrand, -kF, kF)    #fixed_quad sometimes way faster, but at other times way worse
    result = 2*integral + (U-2*lagr)*(r**2)*np.sin(phi)**2 - lagr*(1-(r**2))
    #result = 2*integral + (U*(r**2)*np.sin(phi)**2 - lagr*(1-(r**2)*np.cos(2*phi)))
    return result

#helper function for finding mu when kF != np.pi; returns value for passed on k
def E_tot_kF(k,r,phi,lagr,mu):  
    E_k = lambda k : 1/2*(eps_k(k)+eps_f+lagr
                          - np.sqrt((eps_k(k)-eps_f-lagr)**2 + 4*(V*z(r,phi))**2)) - mu # -mu?
    result = E_k(k)
    return result

#when system not at half filling, at kF the 1st summand of the energy should be 0
def find_mu(k,r,phi,lagr):      
    res = E_tot_kF(k,r,phi,lagr,mu=0)
    return res

def approx_fprime(r,phi,lagr,mu,action,delta,kF):   #numerical derivative taken here with scipy
    x0 = np.array([r,phi,lagr])
    deriv = num_dev(x0, E_tot_approx, delta, kF, mu)    
    if action == 'r':
        return deriv[0]
    if action == 'phi':
        return deriv[1]
    if action == 'lagr':
        return deriv[2]
    print('ERROR: derivative not taken; unclear command')

def boundaries(r,phi,lagr,delta):   #only allow certain value ranges for parameters
    if r[-1] < 0 or r[-1]+delta > 1 or np.isnan(r[-1]):
        r.pop(); r.append(r[-1])    #if not valid value, use the previous one that was still OK
    if phi[-1] < 0 or phi[-1]+delta > np.pi/2 or np.isnan(phi[-1]):
        phi.pop(); phi.append(phi[-1])
    #if lagr[-1] < 0 or lagr[-1]+delta >= U/2+1/2 or np.isnan(lagr[-1]):
    if lagr[-1]-delta <= -U/2-1/2 or lagr[-1]+delta >= U/2+1/2 or np.isnan(lagr[-1]): #for m != 0 allow lagr <0 (?)
        lagr.pop(); lagr.append(lagr[-1])
    return r, phi, lagr

def termination(r,phi,lagr,delta,x_tol,kF,mu):  #termination condition for algorithm
    derivative_lagr = approx_fprime(r,phi,lagr,mu,'lagr',delta,kF)
    derivative_r = approx_fprime(r,phi,lagr,mu,'r',delta,kF)
    derivative_phi = approx_fprime(r,phi,lagr,mu,'phi',delta,kF)
    SUM = np.sqrt(derivative_lagr**2 + derivative_r**2 + derivative_phi**2) #condition based on derivatives at given point
    print(derivative_r,derivative_phi, derivative_lagr, SUM); print()
    #if SUM < x_tol: return True
    #else: return False

#fastens the convergence at half-filling
def Barzilai_Borwein_method(x_old,x_new,r,phi,lagr,delta,mu,kF,W):
    gamma = np.zeros(3); difference = []; function = ['r', 'phi', 'lagr']
    for i in function:
        difference.append(approx_fprime(r[-1],phi[-1],lagr[-1],mu,i,delta,kF)-approx_fprime(r[-2],phi[-2],lagr[-2],mu,i,delta,kF))
    for j in range(len(gamma)-1):
#        if abs(x_new[j] - x_old[j]) < 1e-7:
#            gamma[j] = W[j]
#            continue 
        nominator = ((x_new[j] - x_old[j])*(x_new[j] - x_old[j]))
        denominator = (x_new[j] - x_old[j])*difference[j]
        gamma[j] = nominator/denominator
    gamma[2] = W[2]
    return gamma
    
#def adaptive_step_size(r,phi,lagr,delta,W):
#    diff1 = np.zeros(3); diff2 = np.zeros(3)
#    diff1[0] = abs(r[-3]-r[-2]); diff1[1] = abs(phi[-3]-phi[-2]); diff1[2] = abs(lagr[-3]-lagr[-2])
#    diff2[0] = abs(r[-2]-r[-1]); diff2[1] = abs(phi[-2]-phi[-1]); diff2[2] = abs(lagr[-2]-lagr[-1])
#    for j in range(len(diff1)):
#        if 1e5*diff2[j] < diff1[j]: W[j] *= 1/2
#    return W

#iterative optimization algorithm to find a saddle point for KRSB approach
def gradient_descent_recursion(r_init,phi_init,lagr_init,kF): 
    #step size W
    #W = np.array([0.1,0.1,0.2])    #good for mu = 0
    W = np.array([0.1,0.1,0.005])   #when oscilatting lambda
    x_tol = 0.001                   #termination
    maxit = int(1e4)                #value of max. interations
    index = 0                       #count number of iterations
    delta = 1e-3                    #for derivative
    
    #initialization
    r = []; r.append(r_init); phi = []; phi.append(phi_init); lagr = []; lagr.append(lagr_init);
    
    #given some k value resp. charge density, calculate mu if k != np.pi
    if kF != np.pi/a:
        mu = find_mu(kF,r_init,phi_init,lagr_init)
        print(mu)
    else: mu = 0
    
    for i in range(maxit):  #main loop
        r_old, phi_old, lagr_old = r[-1], phi[-1], lagr[-1]
        x_old = np.array([r_old,phi_old,lagr_old])
        
        derivative_lagr = approx_fprime(r_old,phi_old,lagr_old,mu,'lagr',delta,kF)
        #print(derivative_lagr,'deriv_lagr')
        new_lagr = lagr_old + W[2] * derivative_lagr
        lagr.append(new_lagr)
        
        derivative_r = approx_fprime(r_old,phi_old,lagr_old,mu,'r',delta,kF)
        #print(derivative_r,'deriv_r')
        new_r = r_old - W[0] * derivative_r
        r.append(new_r)
        
        derivative_phi = approx_fprime(r_old,phi_old,lagr_old,mu,'phi',delta,kF)
        #print(derivative_phi,'deriv_phi')        
        new_phi = phi_old - W[1] * derivative_phi
        phi.append(new_phi)

        if i >= 1: r, phi, lagr = boundaries(r,phi,lagr,delta)  #check if values in allowed range
        
        x_new = np.array([r[-1],phi[-1],lagr[-1]])
        print(x_new)
        
        if termination(r[-1],phi[-1],lagr[-1],delta,x_tol,kF,mu) and i > 0: break   #termination condition
        
        #works bad at half-filling
        #if i >= 1: W = Barzilai_Borwein_method(x_old,x_new,r,phi,lagr,delta,mu,kF,W)
        
        #if i >= 1: W = adaptive_step_size(r,phi,lagr,delta,W)
        
        if kF != np.pi/a:
            mu = find_mu(kF,r[-1],phi[-1],lagr[-1])
            print(mu)
        else: mu = 0
        
        index += 1  # +1 for iteration counter
        
    #solution = np.array([r[-1],phi[-1],lagr[-1]])
    solution = np.array([r[-2],phi[-2],lagr[-2]])
    
    return solution, index

'_____________________________________________________________________________'

e_func = lambda x, y : x*np.cos(y)
d_func = lambda x, y : x*np.sin(y)
p_func = lambda x : np.sqrt(1/2*(1-x**2))

#parameter values
t = 1; a = 1; kF = np.pi/a
V = 1*t; U = 1.5*t; eps_f = -U/2


#res = gradient_descent_recursion(0.6,0.6,1e-3,kF)
#res = gradient_descent_recursion(0.6,0.6,U/2+1e-3,kF)    #good for mu = 0 ?
res = gradient_descent_recursion(0.6,np.pi/4,U/2+1e-3,kF)   #good for mu = 0 (U = 0?)

