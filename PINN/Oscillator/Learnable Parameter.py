import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
sys.path.append('C:/Users/mosar/Documents/CAS/Project/Playground/Toymodel')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops import math_ops

tf.keras.backend.clear_session()

np.random.seed(123)
tf.random.set_seed(42)

# NN architecture
def create_NN_model():
    
    inputs = tf.keras.Input(shape=(1,))
    x = tf.keras.layers.Dense(32, activation='tanh')(inputs)
    x = tf.keras.layers.Dense(32, activation='tanh')(x)
    x = tf.keras.layers.Dense(32, activation='tanh')(x)
    output = tf.keras.layers.Dense(1)(x)
    
    neural_net = tf.keras.Model(inputs, output)
    
    return neural_net

# Underdamped case
def exact_solution(t, delta, w0, A0, v0):
    assert delta < w0
    w = np.sqrt(w0**2-delta**2)
    phi = np.arctan(1/w*(v0/A0-delta))
    A = A0/(np.cos(phi))
    u = np.exp(-delta*t)*A*np.cos(phi+w*t)
    return u

########################################################################

# Set-up physical model

endpoint = 1

v0 = 0.0    # initial velocity
A0 = 1.0    # initial amplitude

bc = np.array([A0, v0])   # boundary conditions
t_init = np.array([0.0, 0.0]) # initial times for bcs

#k = 2
#d = 0.2
k = 40
d = 4

# For analytical solution with m = 1
delta = d/2
w0 = np.sqrt(k)

########################################################################

# Hyperparameters

epochs = 20000

optm = tf.keras.optimizers.Adam(learning_rate = 0.001)

# Data loss weight
lambda1, lambda2 = 1e-1, 1e-4

NN = create_NN_model()  # build model

# Additional, learnable variables added to gradient tracking
mu = tf.Variable([0.0], trainable=True, name='mu')

# Training data resp. collocation points
train_t = np.linspace(0, endpoint, 30, endpoint=True)
train_u = exact_solution(train_t, delta, w0, bc[0], bc[1]) 

# Testing data points for NN prediction
testing_t = np.linspace(0, endpoint, 300, endpoint=True)

# Exact result from solution for plotting
true_t_plot = np.linspace(0, endpoint, 100, endpoint=True)
true_u_plot = exact_solution(true_t_plot, delta, w0, bc[0], bc[1])

########################################################################

# Training routine

train_loss_record = []
mu_list = []
bc1_list = []
bc2_list = []

for itr in range(epochs+1):
    
    train_coloc = tf.convert_to_tensor(train_t)
    train_coloc = tf.reshape(train_coloc, [-1,1])
    train_coloc = tf.Variable(train_coloc, name='train_coloc')
    
    #l0ss = []
    
    with tf.GradientTape(persistent=True) as tape:
        
        # Boundary loss
        t_init1 = t_init[0]
        t_init1 = tf.convert_to_tensor(t_init1)
        t_init1 = tf.reshape(t_init1, [1, 1])
        t_init1 = tf.constant(t_init1)
        
        pred_init1 = NN(t_init1)
        bc_loss_1 = math_ops.squared_difference(pred_init1, bc[0])  # t_init1 -> x(t) initial condition
        bc_loss_1 = tf.cast(bc_loss_1, tf.float64)
        
        t_init2 = t_init[1]
        t_init2 = tf.convert_to_tensor(t_init2)
        t_init2 = tf.reshape(t_init2, [1, 1])
        t_init2 = tf.Variable(t_init2, name='t_init2')
        
        pred_init2 = NN(t_init2)
        dfdx_t0 = tape.gradient(pred_init2, t_init2)  # x'(t) at t_init
        
        bc_loss_2 = math_ops.squared_difference(dfdx_t0, bc[1])  # t_init2 -> x'(t) initial condition
        bc_loss_2 = tf.cast(bc_loss_2, tf.float64)
        
        # 1st and 2nd order gradients for co-location pts
        with tf.GradientTape(persistent=True) as tape2:
            x_pred = NN(train_coloc)
            dx_dt = tape2.gradient(x_pred, train_coloc)  # 1st derivative x'(t)
     
        d2x_dt2 = tape2.gradient(dx_dt, train_coloc)  # 2nd derivative x''(t)         
        
        x_pred = tf.cast(x_pred, tf.float64)
        mu_var = tf.cast(mu, tf.float64)
        
        # Learnable parameter mu passed to ODE: d2x_dt2 + k/m*x + d/m*dx_dt
        residual = d2x_dt2 + mu_var*dx_dt + k*x_pred
        
        # Collocation points loss
        ode_loss = tf.reduce_mean(math_ops.square(residual))
        ode_loss = tf.cast(ode_loss, tf.float64)
        
        train_loss = ode_loss + bc_loss_1 + lambda1*bc_loss_2
        
        #l0ss.append([ode_loss.numpy(), bc_loss_1.numpy(), lambda1*bc_loss_2.numpy()])
        
        trainable = NN.trainable_variables
        trainable.append(mu)
        
        grad_w = tape.gradient(train_loss, trainable)
        optm.apply_gradients(zip(grad_w, trainable))    # update the weights using the gradients
        
        # Gradient descent
        mu.assign_sub(grad_w[-1] * 0.001) # descent factor: learning rate of trainable parameter   

    train_loss_record.append(train_loss.numpy()[0])
    mu_list.append(mu.numpy()) 
    bc1_list.append(pred_init1.numpy()[0])
    bc2_list.append(dfdx_t0.numpy()[0])
        

    #if itr in loss_tracker:
    if (itr % 500) == 0:  # plot every n epochs
        print(itr, train_loss_record[-1], mu_list[-1], bc1_list[-1], bc2_list[-1])
        #print(l0ss[-1])
        
        pred_u = NN.predict(testing_t)

        plt.figure(figsize = (10,8))
        plt.plot(train_t, train_u, 'ok', label = 'Train')
        plt.plot(true_t_plot, true_u_plot, '-k',label = 'Exact')
        plt.plot(testing_t, pred_u, '--r', label = 'NN')
        plt.legend(fontsize = 20)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.xlabel('t', fontsize = 20)
        plt.ylabel('u', fontsize = 20)
        plt.title(f"Training step {itr}", fontsize = 20)
        #plt.show()
        plt.savefig(f"Figures/Learnable Parameter/Learn_PINN_training_step_{itr}.png", bbox_inches='tight')


pred_u = NN.predict(testing_t)

plt.figure(figsize = (10,8))
plt.plot(train_t, train_u, 'ok', label = 'Train')
plt.plot(true_t_plot, true_u_plot, '-k',label = 'Exact')
plt.plot(testing_t, pred_u, '--r', label = 'NN')
plt.legend(fontsize = 15)
plt.xlabel('t', fontsize = 15)
plt.ylabel('u', fontsize = 15)
plt.title(f"Model fit, {epochs} epochs", fontsize = 15)
#plt.show()
plt.savefig(f"Figures/Learnable Parameter/Learn_PINN_fit_{epochs}.png", bbox_inches='tight')        
  
plt.figure(figsize = (10,8))
plt.title(r"Training of $\mu$", fontsize = 15)
plt.plot(mu_list, 'r', label="PINN estimate")
plt.hlines(d, 0, len(mu_list), label="True value", color="tab:blue")
plt.legend()
plt.grid()
plt.xlabel("Epoch", fontsize = 15)
plt.ylabel(r"$\mu$", fontsize = 15)
plt.legend(fontsize = 15)
#plt.show()
plt.savefig("Figures/Learnable Parameter/Learn_mu_vs_epoch.png", bbox_inches='tight')

plt.figure(figsize = (10,8))
plt.plot(bc1_list, color='r')
plt.xlabel('Epoch', fontsize = 20)
plt.ylabel(r'$x_0$', fontsize = 20)
plt.grid()
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.title("Initial position", fontsize = 20)
plt.savefig("Figures/Learnable Parameter/Learn_PINN_position.png", bbox_inches='tight')

plt.figure(figsize = (10,8))
plt.plot(bc2_list, color='r')
plt.xlabel('Epoch', fontsize = 20)
plt.ylabel(r'$v_0$', fontsize = 20)
plt.grid()
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.title("Initial velocity", fontsize = 20)
plt.savefig("Figures/Learnable Parameter/Learn_PINN_velocity.png", bbox_inches='tight')

plt.figure(figsize = (10,8))
plt.plot(train_loss_record, color='r')
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Loss', fontsize = 15)
plt.ylim(ymin=0)
plt.grid()
plt.title("Model loss", fontsize = 15)
plt.savefig("Figures/Learnable Parameter/Learn_PINN_loss.png", bbox_inches='tight')