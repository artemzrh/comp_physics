import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import tensorflow as tf
import keras
import random
import matplotlib.pyplot as plt
from Functions import rungekutta4, plot_epochs, plot_loss, rungekutta1

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

def trig_func(y, t):
    return np.cos(2*np.pi*t)

def create_NN_model():
    neural_net = tf.keras.models.Sequential([
        tf.keras.layers.Input((1,)),
        # tanh activation as twice differentiable
        tf.keras.layers.Dense(units = 32, activation = 'tanh'),
        tf.keras.layers.Dense(units = 32, activation = 'tanh'),
        tf.keras.layers.Dense(units = 32, activation = 'tanh'),
        tf.keras.layers.Dense(units = 1)
    ])
    return neural_net

# Physics-informed loss function
def ode_system(t, net):
    t = t.reshape(-1,1)
    t = tf.constant(t, dtype = tf.float32)
    t_0 = tf.zeros((1,1))
    one = tf.ones((1,1))

    # record the operations performed on tensors
    with tf.GradientTape() as tape:
        tape.watch(t)

        u = net(t)
        
        # calculate the gradients of the recorded operations with respect to the specified inputs
        u_t = tape.gradient(u, t)

    ode_loss = u_t - tf.math.cos(2*np.pi*t)
    IC_loss = net(t_0) - one
    
    total_loss = tf.reduce_mean(tf.square(ode_loss)) + tf.reduce_mean(tf.square(IC_loss))

    return total_loss

######################################################################## 

# Set up grid points / observations

#train_t = (np.array([0., 0.025, 0.475, 0.5, 0.525, 0.9, 0.95, 1., 1.05, 1.1, 1.4, 1.45, 1.5, 1.55, 1.6, 1.95, 2.])).reshape(-1, 1)
#train_t = np.arange(0, 2+0.05, 0.05)
train_t = np.arange(0, 2+0.2, 0.2)
#train_t = (np.array([0., 0.1, 0.2, 0.24, 0.3, 0.4, 0.5, 0.7, 0.75, 0.8, 0.99, 1, 1.1, 1.2, 1.27, 1.3, 1.4, 1.5, 1.65, 1.75, 1.8, 1.99])).reshape(-1, 1)
#train_t = 2 * np.random.random_sample((41,))

########################################################################

# Runge-Kutta for comparison

y0 = 1
RK_sol4 = rungekutta4(trig_func, y0, np.sort(train_t))
RK_sol1 = rungekutta1(trig_func, y0, np.sort(train_t))

########################################################################

# Physics-Informed Neural Network

optm = tf.keras.optimizers.Adam(learning_rate = 0.001)

NN_phys = create_NN_model()
NN_phys.summary()

test_t = np.linspace(0, 2, 100, endpoint=True)

train_u = np.sin(2*np.pi*train_t)/(2*np.pi) + 1
true_u = np.sin(2*np.pi*test_t)/(2*np.pi) + 1

train_loss_record = []

epochs = 801

#loss_tracker = np.linspace(0, epochs, 10, dtype=int, endpoint=True)

patience = 200
wait = 0
best = float('inf')

early_stop = 0

# Train PINN
for itr in range(epochs):

    with tf.GradientTape() as tape:
        train_loss = ode_system(train_t, NN_phys)
        train_loss_record.append(train_loss)

        grad_w = tape.gradient(train_loss, NN_phys.trainable_variables)
        optm.apply_gradients(zip(grad_w, NN_phys.trainable_variables))

    #if itr in loss_tracker:
    if (itr % 100) == 0:    
        print(train_loss.numpy())
        plot_epochs(train_t, train_u, test_t, true_u, itr, NN_phys)#, f"PINN_training_itr_{itr}.png")
        
    # early stopping
    wait += 1
    if train_loss.numpy() < best:
        best = train_loss.numpy()
        wait = 0
    if wait >= patience:
        print(f"Stopped at iteration {itr} with loss {train_loss_record[itr]}.")
        early_stop = itr
        break

# Save weights
# NN_phys.save_weights('toymodel_weights')

########################################################################

# Plot results

pred_u = NN_phys.predict(test_t).ravel()

plt.figure(figsize = (10,8))
plt.plot(train_t, train_u, 'ok', label = 'Train')
plt.plot(test_t, true_u, '-k',label = 'True')
#plt.plot(np.sort(train_t), RK_sol4, '--b', label = 'RK4')
#plt.plot(np.sort(train_t), RK_sol1, linestyle = 'dotted', color = 'green', label = 'RK1')
plt.plot(test_t, pred_u, '--r', label = 'NN')
plt.legend(fontsize = 15)
plt.xlabel('t', fontsize = 15)
plt.ylabel('u', fontsize = 15)
if early_stop:
    plt.title(f"Model fit, {early_stop} epochs", fontsize = 15)
else:
    plt.title(f"Model fit, {epochs-1} epochs", fontsize = 15)
plt.show()

plot_loss(train_loss_record)

########################################################################

# Standard Neural Network

NN_std = create_NN_model()
NN_std.summary()

train_loss_record = []
loss_fn = keras.losses.MeanAbsoluteError()

epochs = 801

#loss_tracker = np.linspace(0, epochs, 10, dtype=int, endpoint=True)

patience = 200
wait = 0
best = float('inf')

early_stop = 0

# Train NN
for itr in range(epochs):
    
    with tf.GradientTape() as tape:
        y_pred = NN_std(train_t, training=True) 
        loss_value = loss_fn(train_u, y_pred)
        train_loss_record.append(loss_value)

    grad_w = tape.gradient(loss_value, NN_std.trainable_variables)
    optm.apply_gradients(zip(grad_w, NN_std.trainable_variables))

    #if itr in loss_tracker:
    if (itr % 100) == 0:  
        print(loss_value.numpy())
        plot_epochs(train_t, train_u, test_t, true_u, itr, NN_std)#, f"NN_training_itr_{itr}.png")
        
    # early stopping
    # wait += 1
    # if loss_value.numpy() < best:
    #     best = loss_value.numpy()
    #     wait = 0
    # if wait >= patience:
    #     print(f"Stopped at iteration {itr} with loss {train_loss_record[itr]}.")
    #     early_stop = itr
    #     break

########################################################################

# Plot results

test_t = np.linspace(0, 2, 100)

train_u = np.sin(2*np.pi*train_t)/(2*np.pi) + 1
true_u = np.sin(2*np.pi*test_t)/(2*np.pi) + 1
pred_u = NN_std.predict(test_t).ravel()

plt.figure(figsize = (10,8))
plt.plot(train_t, train_u, 'ok', label = 'Train')
plt.plot(test_t, true_u, '-k',label = 'True')
#plt.plot(np.sort(train_t), RK_sol4, '--b', label = 'RK4')
#plt.plot(np.sort(train_t), RK_sol1, linestyle = 'dotted', color = 'green', label = 'RK1')
plt.plot(test_t, pred_u, '--r', label = 'NN')
plt.legend(fontsize = 15)
plt.xlabel('t', fontsize = 15)
plt.ylabel('u', fontsize = 15)
if early_stop:
    plt.title(f"Model fit, {early_stop} epochs", fontsize = 15)
else:
    plt.title(f"Model fit, {epochs-1} epochs", fontsize = 15)
plt.show()

plot_loss(train_loss_record)