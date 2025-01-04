# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:27:55 2020

@author: Artem
"""

import numpy as np
import scipy.sparse as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#input two matrices A, B; block diagonal matrix as return
def diagonal_combine(A,B,boundary):
    size = 2*N
    CDiag = np.block([[A, np.zeros((size, size))],[np.zeros((size, size)), B]])
    CDiag[A.shape[0]-2][A.shape[1]] = -t    #connetion of c's of tb and H_V system
    CDiag[A.shape[0]][A.shape[1]-2] = -t    # -"- position HIGHLY important!
    #CDiag[A.shape[0]-1][A.shape[1]+1] = epsilon_f_V    #connetion of f's of tb and H_V system
    #CDiag[A.shape[0]+1][A.shape[1]-1] = epsilon_f_V    # -"- not needed!
    if boundary == True:                    #periodic boundary conditions if necessary
        CDiag[0][CDiag.shape[1]-2] = -t     #was CDiag.shape[1]-1 17.07.20 16:42
        CDiag[CDiag.shape[0]-2][0] = -t
    return(CDiag)

def to_plot(a,b):                       #connects two arrays into one; for plotting needed
    return(np.concatenate([a,b]))

############################################################## set-up

#choice of parameter values
N = 50
t = 1
a = 1
mu = 0
V = 0.3*t       # = 1 zu gross im Vgl. zu t
epsilon_f_tb = 100
epsilon_f_V = -0/2

#hybridized system double the size of tb system to get equal amount of c states in
    #both systems
d = np.zeros(2*N) # diagonals
for i in range(2*N):
    if (i % 2) == 0: d[i] = -mu     # alternating (c,f), such that c*c starts
    else: d[i] = epsilon_f_V

d1 = np.zeros(2*N-1) # hybdridization of c, f, therefore will be on one off-diagonal
for i in range(2*N-1):
    if (i % 2) == 0: d1[i] = V

    
d2 = np.zeros(2*N-2) # hopping in hybrid. system, therefore will be on two off-diagonal
for i in range(2*N-2):
    if (i % 2) == 0: d2[i] = -t
    
hybrid_open = sc.diags([d, d1, d2, d1, d2],     #construct a sparse matrix from diagonals
                             [0, 1, 2, -1, -2], format='csc')
#print(hybrid_open.toarray())

#extension of tb system to plot its "f-states" as well; reduces computational effort too
d = np.zeros(2*N) # diagonals tb system
for i in range(2*N):
    if (i % 2) == 0: d[i] = -mu #-2*t*np.cos(k)
    else: d[i] = epsilon_f_tb
    
d2 = np.zeros(2*N-2) # hopping; on one off-diagonal
for i in range(2*N-2):
    if (i % 2) == 0: d2[i] = -t

#construct a sparse matrix from diagonals
hopping_open = sc.diags([d, d2, d2], [0, 2, -2], format='csc')
#print(hopping_open.toarray())

#combine two matrices to block diagonal matrix 
combination_per = diagonal_combine(hopping_open.toarray(),hybrid_open.toarray(), True)
combination_open = diagonal_combine(hopping_open.toarray(),hybrid_open.toarray(), False)
#print(combination_per)

#diagonalization - eigenvalues and eigenvectors
vals_per, eigv_per = np.linalg.eigh(combination_per)
#cut out f-energies of tb system from plot
position = (np.where(np.around(vals_per) == np.around(vals_per).max()))[0][0]

#plot spectrum
x = np.arange(0, position, 1)
plt.plot(x, vals_per[:position], 'r'); plt.title(r'$H_{tb}+H_V \,$'+'| per bc')
plt.text(0, 1, r'$t = {}$'.format(t)+'\n'+r'$\mu={}$'.format(mu)+'\n'+r'$V={}$'.format(V)+'\n'+r'$\epsilon_f={}$'.format(epsilon_f_V),
         ha='left', va='top',)
plt.xlabel('eigenvalue j | '+r'$k \in [0,\frac{\pi}{a}]$'+' with '+r'$a = 1$')
plt.ylabel('energies'); plt.grid(True); plt.savefig('spectrum, per bc.png', bbox_inches='tight')
plt.show()

###########################################

#diagonalization - eigenvalues and eigenvectors
vals_open, eigv_open = np.linalg.eigh(combination_open)
#cut out f-energies of tb system from plot
position = (np.where(np.around(vals_open) == np.around(vals_open).max()))[0][0]

#plot spectrum
x = np.arange(0, position, 1)
plt.plot(x, vals_open[:position], 'r'); plt.title(r'$H_{tb}+H_V \,$'+'| open bc')
plt.text(0, 1, r'$t = {}$'.format(t)+'\n'+r'$\mu={}$'.format(mu)+'\n'+r'$V={}$'.format(V)+'\n'+r'$\epsilon_f={}$'.format(epsilon_f_V),
         ha='left', va='top',)
plt.xlabel('eigenvalue j | '+r'$k \in [0,\frac{\pi}{a}]$'+' with '+r'$a = 1$')
plt.ylabel('energies'); plt.grid(True); plt.savefig('spectrum, open bc.png', bbox_inches='tight')
plt.show()      


#plotting eigenvectors to eigenvalue i=a_plot
a_plot = 10

#x1 = np.arange(0, N, 1)
x = np.arange(0, 2*N, 1)

f = eigv_per[1::2,a_plot]
c = eigv_per[::2,a_plot]

plt.plot(x, f, 'b-', label='f')
plt.plot(x, c, 'r-', label='c')
plt.title(r'$c_i$'+' eigenvectors | per bc')
plt.grid(True); plt.legend(loc='best', frameon=False)
plt.savefig('eigenvectors, '+r'j={}'.format(a_plot)+ ', per bc.png', bbox_inches='tight'); plt.show()


f = eigv_open[1::2,a_plot]
c = eigv_open[::2,a_plot]

plt.plot(x, f, 'b-', label='f')
plt.plot(x, c, 'r-', label='c')
plt.title(r'$c_i$'+' eigenvectors | open bc')
plt.grid(True); plt.legend(loc='best', frameon=False)
plt.savefig('eigenvectors, '+r'j={}'.format(a_plot)+ ', open bc.png', bbox_inches='tight'); plt.show()


############################################################## spectral function etc.

upper = -min(vals_per)
#E = np.arange(min(vals_per), upper, (upper-min(vals_per))/N)
E = np.arange(min(vals_open), upper, (upper-min(vals_open))/(4*N))

#delta = 0.0455 #should be smaller than 0.15, such that "delta" peaks emerge
delta = 0.05

A = np.zeros((E.size, 2*N)); summ = 0;  #initialization of A(x,E)
help_arr = np.arange(0,vals_per.size,1);        #to handle eigenvectors of hybridized system
s = 0

# c's periodic b.c.
for i in range(2*N):    #positions
    if i%int(N/2) == 0: print(i, s, help_arr[s])    #to know how far loop is
    for j in range(E.size):     #energies
        for l in range(vals_per.size):  #summation over eigenvalues for specific (x,E)
            #every second elements is a c
            summ+= delta/(2*np.pi)*(eigv_per[help_arr[s],l]*eigv_per[help_arr[s],l])/((E[j]-vals_per[l])**2+delta**2)
        A[j][i] = summ; summ = 0    #saving the computed value into A(x,E)
    s+=2;    #to pick out correct c's in hybrid. system, jump two further

#plot contour plot of spectral function
X, Y = np.meshgrid(np.arange(0, 2*N, 1), E)
Z = np.asarray(A)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z, cmap='jet')
fig.colorbar(cp)
ax.set_title('Contour Plot - Spectr Function of '+r'$c_i$'+' (per bc)')
ax.set_xlabel(r'$\mathdefault{x_{i} = i \cdot a}$')
ax.set_ylabel('E')
plt.savefig('spectral function contour c, per bc.png', bbox_inches='tight'); plt.show()

# Plot 3D surface
X, Y = np.meshgrid(np.arange(0, 2*N, 1), E)
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_title('3D Plot Spectr Function of '+r'$c_i$'+' (per bc)'); ax1.set_xlabel(r'$\mathdefault{x_{i} = i \cdot a}$')
ax1.set_ylabel('E'); ax1.set_zlabel(r'$A(x_{i},E)$')
mycmap = plt.get_cmap('jet')
surf1 = ax1.plot_surface(X, Y, Z[:,:], cmap=mycmap)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
ax1.view_init(azim=-30) # rotate the axes
plt.savefig('spectral function 3D c, per bc.png', bbox_inches='tight'); plt.show()

#plot specific interface of spectral function
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(E, Z[:,N-9], linestyle='solid', color='red', label='N-9')
ax1.plot(E, Z[:,N-8], linestyle='solid', color='blue', label='N-8')
ax1.plot(E, Z[:,N-7], linestyle='solid', color='black', label='N-7')
ax2.plot(E, Z[:,N-6], linestyle='solid', color='red', label='N-6')
ax2.plot(E, Z[:,N-5], linestyle='solid', color='blue', label='N-5')
ax2.plot(E, Z[:,N-4], linestyle='solid', color='black', label='N-4')
plt.suptitle(r'$A(x_{i},E)$'+' for '+r'$c_i$'+' around N | per bc')
ax1.legend(loc='best', frameon=False); ax2.legend(loc='best', frameon=False);
ax1.grid(True); ax2.grid(True);
plt.savefig('spectral function interface1 c, per bc.png', bbox_inches='tight'); plt.show()

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(E, Z[:,N-3], linestyle='solid', color='red', label='N-3')
ax1.plot(E, Z[:,N-2], linestyle='solid', color='blue', label='N-2')
ax1.plot(E, Z[:,N-1], linestyle='solid', color='black', label='N-1')
ax2.plot(E, Z[:,N], linestyle='solid', color='red', label='N')
ax2.plot(E, Z[:,N+1], linestyle='solid', color='blue', label='N+1')
ax2.plot(E, Z[:,N+2], linestyle='solid', color='black', label='N+2')
plt.suptitle(r'$A(x_{i},E)$'+' for '+r'$c_i$'+' around N | per bc')
ax1.legend(loc='best', frameon=False); ax2.legend(loc='best', frameon=False);
ax1.grid(True); ax2.grid(True); 
plt.savefig('spectral function interface2 c, per bc.png', bbox_inches='tight'); plt.show()

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(E, Z[:,N+3], linestyle='solid', color='red', label='N+3')
ax1.plot(E, Z[:,N+4], linestyle='solid', color='blue', label='N+4')
ax1.plot(E, Z[:,N+5], linestyle='solid', color='black', label='N+5')
ax2.plot(E, Z[:,N+6], linestyle='solid', color='red', label='N+6')
ax2.plot(E, Z[:,N+7], linestyle='solid', color='blue', label='N+7')
ax2.plot(E, Z[:,N+8], linestyle='solid', color='black', label='N+8')
plt.suptitle(r'$A(x_{i},E)$'+' for '+r'$c_i$'+' around N | per bc')
ax1.legend(loc='best', frameon=False); ax2.legend(loc='best', frameon=False);
ax1.grid(True); ax2.grid(True); 
plt.savefig('spectral function interface3 c, per bc.png', bbox_inches='tight'); plt.show()

########## f's periodic b.c.
#computation cf. above for c-case; only difference is help_arr[s]+1 to extract f's

start = N-5
A = np.zeros((E.size, (2*N-start))); summ = 0; s = 0
s = 2*start

for i in range(2*N-start):
    if i%int(N/2) == 0: print(i, s, help_arr[s]+1)
    for j in range(E.size):
        for l in range(vals_per.size):
            summ+= delta/(2*np.pi)*(eigv_per[help_arr[s]+1,l]*eigv_per[help_arr[s]+1,l])/((E[j]-vals_per[l])**2+delta**2)
        A[j][i] = summ; summ = 0
    s+=2; 

X, Y = np.meshgrid(np.arange(start, 2*N, 1), E)
Z = np.asarray(A)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z, cmap='jet')
fig.colorbar(cp)
ax.set_title('Contour Plot - Spectr Function of '+r'$f_i$'+' (per bc)')
ax.set_xlabel(r'$\mathdefault{x_{i} = i \cdot a}$')
ax.set_ylabel('E')
plt.savefig('spectral function contour f, per bc.png', bbox_inches='tight'); plt.show()

# Plot 3D surface
X, Y = np.meshgrid(np.arange(start, 2*N, 1), E)
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_title('3D Plot Spectr Function of '+r'$f_i$'+' (per bc)'); ax1.set_xlabel(r'$\mathdefault{x_{i} = i \cdot a}$')
ax1.set_ylabel('E'); ax1.set_zlabel(r'$A(x_{i},E)$')
mycmap = plt.get_cmap('jet')
surf1 = ax1.plot_surface(X, Y, Z[:,:], cmap=mycmap)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
ax1.view_init(azim=-110) # rotate the axes
plt.savefig('spectral function 3D f, per bc.png', bbox_inches='tight'); plt.show()

plt.plot(E, Z[:,4], 'y-', label='N-1')
plt.plot(E, Z[:,5], 'b-', label='N')
plt.plot(E, Z[:,6], 'r-', label='N+1')
plt.plot(E, Z[:,int(N/2+5)], 'g-', label='N+50')
plt.title(r'$A(x_{i},E)$'+' for '+r'$f_i$'+' around N | per bc')
plt.grid(True); plt.legend(loc='best', frameon=False)
plt.savefig('spectral function interface f, per bc.png', bbox_inches='tight'); plt.show()


############################ open boundary conditions

#delta = 0.0355
delta = 0.055

upper = -min(vals_open)
#E = np.arange(min(vals_open), upper, (upper-min(vals_open))/N)
E = np.arange(min(vals_open), upper, (upper-min(vals_open))/(4*N))

A = np.zeros((E.size, 2*N)); summ = 0;
s = 0

# c's open b.c.
for i in range(2*N):
    if i%int(N/2) == 0: print(i, s, help_arr[s])
    for j in range(E.size):
        for l in range(vals_open.size):
            summ+= delta/(2*np.pi)*(eigv_open[help_arr[s],l]*eigv_open[help_arr[s],l])/((E[j]-vals_open[l])**2+delta**2)
        A[j][i] = summ; summ = 0
    s+=2; 


X, Y = np.meshgrid(np.arange(0, 2*N, 1), E)
Z = np.asarray(A)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z, cmap='jet')
fig.colorbar(cp)
ax.set_title('Contour Plot - Spectr Function of '+r'$c_i$'+' (open bc)')
ax.set_xlabel(r'$\mathdefault{x_{i} = i \cdot a}$')
ax.set_ylabel('E')
plt.savefig('spectral function contour c, open bc.png', bbox_inches='tight'); plt.show()

# Plot 3D surface
X, Y = np.meshgrid(np.arange(0, 2*N, 1), E)
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_title('3D Plot Spectr Function of '+r'$c_i$'+' (open bc)'); ax1.set_xlabel(r'$\mathdefault{x_{i} = i \cdot a}$')
ax1.set_ylabel('E'); ax1.set_zlabel(r'$A(x_{i},E)$')
mycmap = plt.get_cmap('jet')
surf1 = ax1.plot_surface(X, Y, Z[:,:], cmap=mycmap)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
ax1.view_init(azim=-30) # rotate the axes
plt.savefig('spectral function 3D c, open bc.png', bbox_inches='tight'); plt.show()

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(E, Z[:,N-9], linestyle='solid', color='red', label='N-9')
ax1.plot(E, Z[:,N-8], linestyle='solid', color='blue', label='N-8')
ax1.plot(E, Z[:,N-7], linestyle='solid', color='black', label='N-7')
ax2.plot(E, Z[:,N-6], linestyle='solid', color='red', label='N-6')
ax2.plot(E, Z[:,N-5], linestyle='solid', color='blue', label='N-5')
ax2.plot(E, Z[:,N-4], linestyle='solid', color='black', label='N-4')
plt.suptitle(r'$A(x_{i},E)$'+' for '+r'$c_i$'+' around N | open bc')
ax1.legend(loc='best', frameon=False); ax2.legend(loc='best', frameon=False);
ax1.grid(True); ax2.grid(True);
plt.savefig('spectral function interface1 c, open bc.png', bbox_inches='tight'); plt.show()

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(E, Z[:,N-3], linestyle='solid', color='red', label='N-3')
ax1.plot(E, Z[:,N-2], linestyle='solid', color='blue', label='N-2')
ax1.plot(E, Z[:,N-1], linestyle='solid', color='black', label='N-1')
ax2.plot(E, Z[:,N], linestyle='solid', color='red', label='N')
ax2.plot(E, Z[:,N+1], linestyle='solid', color='blue', label='N+1')
ax2.plot(E, Z[:,N+2], linestyle='solid', color='black', label='N+2')
plt.suptitle(r'$A(x_{i},E)$'+' for '+r'$c_i$'+' around N | open bc')
ax1.legend(loc='best', frameon=False); ax2.legend(loc='best', frameon=False);
ax1.grid(True); ax2.grid(True); 
plt.savefig('spectral function interface2 c, open bc.png', bbox_inches='tight'); plt.show()

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(E, Z[:,N+3], linestyle='solid', color='red', label='N+3')
ax1.plot(E, Z[:,N+4], linestyle='solid', color='blue', label='N+4')
ax1.plot(E, Z[:,N+5], linestyle='solid', color='black', label='N+5')
ax2.plot(E, Z[:,N+6], linestyle='solid', color='red', label='N+6')
ax2.plot(E, Z[:,N+7], linestyle='solid', color='blue', label='N+7')
ax2.plot(E, Z[:,N+8], linestyle='solid', color='black', label='N+8')
plt.suptitle(r'$A(x_{i},E)$'+' for '+r'$c_i$'+' around N | open bc')
ax1.legend(loc='best', frameon=False); ax2.legend(loc='best', frameon=False);
ax1.grid(True); ax2.grid(True); 
plt.savefig('spectral function interface3 c, open bc.png', bbox_inches='tight'); plt.show()

########## f's open b.c.

start = N-5
A = np.zeros((E.size, (2*N-start))); summ = 0; s = 0
s = 2*start

for i in range(2*N-start):
    if i%int(N/2) == 0: print(i, s, help_arr[s])
    for j in range(E.size):
        for l in range(vals_open.size):
            summ+= delta/(2*np.pi)*(eigv_open[help_arr[s]+1,l]*eigv_open[help_arr[s]+1,l])/((E[j]-vals_open[l])**2+delta**2)
        A[j][i] = summ; summ = 0
    s+=2; 

X, Y = np.meshgrid(np.arange(start, 2*N, 1), E)
Z = np.asarray(A)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z, cmap='jet')
fig.colorbar(cp)
ax.set_title('Contour Plot - Spectr Function of '+r'$f_i$'+' (open bc)')
ax.set_xlabel(r'$\mathdefault{x_{i} = i \cdot a}$')
ax.set_ylabel('E')
plt.savefig('spectral function contour f, open bc.png', bbox_inches='tight'); plt.show()

# Plot 3D surface
X, Y = np.meshgrid(np.arange(start, 2*N, 1), E)
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_title('3D Plot Spectr Function of '+r'$f_i$'+' (open bc)'); ax1.set_xlabel(r'$\mathdefault{x_{i} = i \cdot a}$')
ax1.set_ylabel('E'); ax1.set_zlabel(r'$A(x_{i},E)$')
mycmap = plt.get_cmap('jet')
surf1 = ax1.plot_surface(X, Y, Z[:,:], cmap=mycmap)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
ax1.view_init(azim=-110) # rotate the axes
plt.savefig('spectral function 3D f, open bc.png', bbox_inches='tight'); plt.show()

plt.plot(E, Z[:,4], 'y-', label='N-1')
plt.plot(E, Z[:,5], 'b-', label='N')
plt.plot(E, Z[:,6], 'r-', label='N+1')
plt.plot(E, Z[:,int(N/2+5)], 'g-', label='N+50')
plt.title(r'$A(x_{i},E)$'+' for '+r'$f_i$'+' around N | open bc')
plt.grid(True); plt.legend(loc='best', frameon=False)
plt.savefig('spectral function interface f, open bc.png', bbox_inches='tight'); plt.show()
