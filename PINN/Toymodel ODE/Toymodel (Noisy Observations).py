import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import tensorflow as tf
import random
from Functions import (rungekutta4, plot_figs_with_noise, create_NN_model,
                            train_NN_data_loss, plot_loss, impulse_noise)

def trig_func(y, t):
    return np.cos(2*np.pi*t)

func = lambda t: tf.math.cos(2*np.pi*t)

bc = tf.ones((1,1))

########################################################################

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# Training data

#train_t = 2 * np.random.random_sample((41,))
#train_t = np.array([0., 0.1, 0.2, 0.24, 0.3, 0.4, 0.5, 0.7, 0.75, 0.8, 0.99, 1, 1.1, 1.2, 1.27, 1.3, 1.4, 1.5, 1.65, 1.75, 1.8, 1.99])
#train_t = np.arange(0, 2+0.2, 0.2)
train_t = np.linspace(0, 2, 10, endpoint=True)
#train_t = np.arange(0, 2+0.1, 0.1)

#train_t = np.arange(0, 2+0.05, 0.05)
train_u = np.sin(2*np.pi*train_t)/(2*np.pi) + 1

# True result for plotting
test_t_plot = np.linspace(0, 2, 100, endpoint=True)
true_u_plot = np.sin(2*np.pi*test_t_plot)/(2*np.pi) + 1

# Data for adding noise
# IMPORTANT: the amount of data collocation points has to be equal to physics collocation points
# otherwise the loss computation does not work
data_t = np.sort(2 * np.random.random_sample((10,)))
data_u = np.sin(2*np.pi*data_t)/(2*np.pi) + 1

# Testing data points for NN prediction
testing_t = np.linspace(0, 2, 100, endpoint=True)

# Introduce noise to observations
#data_u_noised = impulse_noise(data_u)
data_u_noised = data_u + np.random.normal(loc=0.0, scale=0.04, size=data_t.shape)

# RK4 solution, although not really considerable for a comparison, because RK cannot take noisy data as base points
y0 = 1
RK_sol = rungekutta4(trig_func, y0, np.sort(np.concatenate((train_t, data_t))))

########################################################################

# Untrained NN

optm = tf.keras.optimizers.Adam(learning_rate = 0.001)

# hyperparameter for data loss weight
# 1e4 too strong weight for the data loss
lambda1 = [10, 1, 0.1, 0.01, 0.001, 0.0001]

final_loss = []

# analyze accuracy of fit w.r.t. lambda1
for param in lambda1:
    
    print("Lambda: ", param)
    NN = create_NN_model()
    
    epochs = 2000
    
    # Train NN
    train_loss_record, early_break = train_NN_data_loss(epochs, optm, NN, func, bc, param,
                                                        train_t, train_u, data_t, data_u,
                                                        data_u_noised, test_t_plot, true_u_plot,
                                                        testing_t)
    
    #pred_u_untrained = NN(data_t)
    pred_u_untrained = NN(testing_t)
    
    if early_break:
        epochs = early_break
    
    plot_figs_with_noise(train_t, train_u, data_t, data_u_noised, test_t_plot, true_u_plot, RK_sol, testing_t, pred_u_untrained, epochs)
    
    plot_loss(train_loss_record)
    
    final_loss.append(train_loss_record[-1].numpy())
