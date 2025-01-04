# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 08:29:39 2018

@author: Artem
"""

import numpy as np
import scipy as sc
from sympy import Matrix
import matplotlib.pyplot as plt

##########################################################################################

N = 81 #only uneven numbers work with the plot of the odd states!!!
max_D = 512 #has to be a proper devisor of 4 !!!
Lambda = 2.5
U_coulomb_repulsion = 0.001
epsilon_electron = np.zeros(1000)
epsilon_impurity = -0.0005
V = 0.004

f_dagger_up = [[0., 0., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 1., 0.]]
f_up = [[0., 1., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 0., 0.]]
f_dagger_down = [[0., 0., 0., 0.], [0., 0., 0., 0.], [1., 0., 0., 0.], [0., 1., 0., 0.]]
f_down = [[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.]]

c_dagger_up = f_dagger_up; c_up = f_up; c_dagger_down = f_dagger_down; c_down = f_down

###########################################################################################

m1 = Matrix(f_dagger_up); m2 = Matrix(c_up); tensor_fdu_cu = np.kron(m1, m2)
    
#m1 = Matrix(c_dagger_up); m2 = Matrix(f_up); tensor_cdu_fu = np.kron(m1, m2)
    
m1 = Matrix(f_dagger_down); m2 = Matrix(c_down); tensor_fdd_cd = np.kron(m1, m2)
    
m1 = Matrix(c_dagger_down); m2 = Matrix(f_down); tensor_cdd_fd = np.kron(m1, m2)

matrix_cdd_cd = np.dot(c_dagger_down, c_down); tensor_cdd_cd = np.kron(np.eye(4), matrix_cdd_cd)
    
matrix_cdu_cu = np.dot(c_dagger_up, c_up); tensor_cdu_cu = np.kron(np.eye(4), matrix_cdu_cu)

matrix_fdu_fu = np.dot(f_dagger_up, f_up); tensor_fdu_fu = np.kron(matrix_fdu_fu, np.eye(4))
    
matrix_fdd_fd = np.dot(f_dagger_down, f_down); tensor_fdd_fd = np.kron(matrix_fdd_fd, np.eye(4))

temp = np.dot(f_dagger_up, f_up); temp2 = np.dot(f_dagger_down, f_down)
double_density_matrix_4x4 = np.dot(temp, temp2)
double_density_matrix = np.kron(double_density_matrix_4x4, np.eye(4))


diagonal_matrix = (epsilon_impurity*(tensor_fdu_fu + tensor_fdd_fd)
                    + U_coulomb_repulsion*double_density_matrix)

H_zero = np.power(Lambda, -1/2)*(diagonal_matrix + V*(tensor_fdu_cu + tensor_fdu_cu.transpose())
        + epsilon_electron[0]*(tensor_cdu_cu + tensor_cdd_cd)
        + V*(tensor_fdd_cd + tensor_fdd_cd.transpose() + tensor_fdu_cu + tensor_fdu_cu.transpose() ))

H_beginning = sc.sparse.coo_matrix((int(np.sqrt(np.size(H_zero))),
                                       int(np.sqrt(np.size(H_zero))))).toarray()


for i in range(int(np.sqrt(np.size(H_beginning)))):     #converting formats
    for j in range(int(np.sqrt(np.size(H_beginning)))):
        H_beginning[i][j]= H_zero[i][j]

############################################################################################

NRG_lowest_energies = np.zeros([10, 1])
iteration_matrix = H_beginning


for i in range(1, N+2):     #the core of the code
    t_N = np.power(Lambda, -0.5*i)
    eigenvalues, eigenvectors = np.linalg.eigh(iteration_matrix)
    eigenvalues = eigenvalues*np.power(Lambda, 1/2)
    eigenvalues = eigenvalues - eigenvalues[0]
    U = eigenvectors
    U_T = np.transpose(eigenvectors)
    
    z = np.zeros((10,1))    #initialisation
    z[0][0] = eigenvalues[0]; z[1][0] = eigenvalues[1]; z[2][0] = eigenvalues[2]
    z[3][0] = eigenvalues[3]; z[4][0] = eigenvalues[4]; z[5][0] = eigenvalues[5]
    z[6][0] = eigenvalues[6]; z[7][0] = eigenvalues[1]; z[8][0] = eigenvalues[2]
    z[9][0] = eigenvalues[9]; NRG_lowest_energies = np.hstack((NRG_lowest_energies, z))
    
    
#    print(NRG_lowest_energies)
    print(i)
    
    iteration_matrix = np.diag(eigenvalues)
    
    XY = int(iteration_matrix.shape[0]/4)
    
    if(iteration_matrix.shape[0] > max_D):    #cut-off
        iteration_matrix = iteration_matrix[:max_D, :max_D]
    
    if(iteration_matrix.shape[0] >= max_D):
        XYZ = int(max_D)
    else:
        XYZ = int(np.power(4, i+1))
        
#    print(i, XY, XYZ)
    
    c_diagonal_N_up = np.kron(np.eye(XYZ), epsilon_electron[i+1]*np.dot(c_dagger_up, c_up))
    c_diagonal_N_down = np.kron(np.eye(XYZ), epsilon_electron[i+1]*np.dot(c_dagger_down, c_down))
    
#    print('Test1')
    
    interm_exchange_1_up = np.dot(np.dot(U_T, np.kron(np.eye(XY), c_dagger_up)), U)
    interm_exchange_1_down = np.dot(np.dot(U_T, np.kron(np.eye(XY), c_dagger_down)), U)
    interm_exchange_2_up = np.dot(np.dot(U_T, np.kron(np.eye(XY), c_up)), U)
    interm_exchange_2_down = np.dot(np.dot(U_T, np.kron(np.eye(XY), c_down)), U)

#    print('Test2')    
    
#    print(interm_exchange_1_up.shape[0])
    
    if(interm_exchange_1_up.shape[0] > max_D): interm_exchange_1_up = interm_exchange_1_up[:max_D, :max_D]
    if(interm_exchange_1_down.shape[0] > max_D): interm_exchange_1_down = interm_exchange_1_down[:max_D, :max_D]
    if(interm_exchange_2_up.shape[0] > max_D): interm_exchange_2_up = interm_exchange_2_up[:max_D, :max_D]
    if(interm_exchange_2_down.shape[0] > max_D): interm_exchange_2_down = interm_exchange_2_down[:max_D, :max_D]
    
#    print(interm_exchange_1_up.shape[0])
    
    c_exchange = t_N*(
        np.kron(interm_exchange_1_up, c_up)
      + np.kron(interm_exchange_2_up, c_dagger_up)
      + np.kron(interm_exchange_1_down, c_down)
      + np.kron(interm_exchange_2_down, c_dagger_down)  
        )
    
#    print('Test3')
    
#    print(np.shape(iteration_matrix), np.shape(c_diagonal_N_up), np.shape(c_exchange))
    
    iteration_matrix = (np.kron(iteration_matrix, np.eye(4))
                        + np.power(Lambda,i/2)*(c_diagonal_N_up + c_diagonal_N_down)
                        + np.power(Lambda,i/2)*c_exchange)


##########################################################################################

n = np.arange(1, N+2, 1)    #complete plot
fig, ax = plt.subplots()        
ax.set_color_cycle(['red', 'black', 'yellow', 'blue', 'gray'])
plt.plot(n, NRG_lowest_energies[0,1:], label='1st')
plt.plot(n, NRG_lowest_energies[1,1:], label='2nd')
plt.plot(n, NRG_lowest_energies[2,1:], label='3rd')
plt.plot(n, NRG_lowest_energies[3,1:], label='4th')
plt.plot(n, NRG_lowest_energies[4,1:], label='5th')
plt.plot(n, NRG_lowest_energies[5,1:], label='6th')
plt.plot(n, NRG_lowest_energies[6,1:], label='7th')
plt.plot(n, NRG_lowest_energies[7,1:], label='8th')
plt.plot(n, NRG_lowest_energies[8,1:], label='9th')
plt.plot(n, NRG_lowest_energies[9,1:], label='10th')
plt.grid(True)
plt.xlabel('N')
plt.ylabel('lowest energy states')
plt.title('NRG flow')
plt.legend()
plt.axis('auto')
plt.show()

n = np.arange(1, (N//2)+2, 1)   #plotting odd sites
#print(n, NRG_lowest_energies[2,2::2])
fig, ax = plt.subplots()        
ax.set_color_cycle(['red', 'black', 'yellow', 'blue', 'gray'])
plt.plot(n, NRG_lowest_energies[0,2::2], label='1st')
plt.plot(n, NRG_lowest_energies[1,2::2], label='2nd')
plt.plot(n, NRG_lowest_energies[2,2::2], label='3rd')
plt.plot(n, NRG_lowest_energies[3,2::2], label='4th')
plt.plot(n, NRG_lowest_energies[4,2::2], label='5th')
plt.plot(n, NRG_lowest_energies[5,2::2], label='6th')
plt.plot(n, NRG_lowest_energies[6,2::2], label='7th')
plt.plot(n, NRG_lowest_energies[7,2::2], label='8th')
plt.plot(n, NRG_lowest_energies[8,2::2], label='9th')
plt.plot(n, NRG_lowest_energies[9,2::2], label='10th')
plt.grid(True)
plt.xlabel('N')
plt.ylabel('lowest energy states')
plt.title('NRG flow (odd)')
plt.legend()
plt.axis('auto')
plt.show()

np.savetxt('energy.out', NRG_lowest_energies, newline='\n')