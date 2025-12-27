import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from Functions import rungekutta4, create_NN_model, train_NN, plot_loss, rungekutta1

def trig_func(y, t):
    return np.cos(2*np.pi*t)

########################################################################

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

y0 = 1

optm = tf.keras.optimizers.Adam(learning_rate = 0.001)

NN = create_NN_model()

# Load the previously saved weights for train_t = np.arange(0, 2+0.05, 0.05)
# "transfer learning" from "Toymodel ODE.py"
NN.load_weights('toymodel_weights')

NN.summary()

# Needed for model.evaluate comand
NN.compile(optm, loss=tf.keras.losses.MSE)

# TRANSFER LEARNING
# Generalize to new grid points -> does not need extra training for this

train_tight1 = np.linspace(0, 2, 200, endpoint=True)
train_tight2 = np.linspace(0, 2, 40, endpoint=True)
train_wide = np.linspace(0, 2, 10, endpoint=True)
train_adapt = (np.array([0., 0.1, 0.2, 0.24, 0.3, 0.4, 0.5, 0.7, 0.75, 0.8, 0.99, 1, 1.1, 1.2, 1.27, 1.3, 1.4, 1.5, 1.65, 1.75, 1.8, 1.99])).reshape(-1, 1)
train_rand = np.sort(2 * np.random.random_sample((22,)))

# Compare NN to ODE solvers w.r.t. collocation points

collocation_list = [train_tight1, train_tight2, train_wide, train_rand]

for itr in range(len(collocation_list)):

    train_t = collocation_list[itr]  

    RK_sol4 = rungekutta4(trig_func, y0, np.sort(train_t))
    RK_sol1 = rungekutta1(trig_func, y0, np.sort(train_t))
    
    test_t = np.linspace(0, 2, 100, endpoint=True)
    
    train_u = np.sin(2*np.pi*train_t)/(2*np.pi) + 1
    true_u = np.sin(2*np.pi*test_t)/(2*np.pi) + 1
    pred_u = NN.predict(train_t).ravel()
    
    #print(np.square(RK_sol1 - train_u))
    loss_RK1 = np.mean(np.square(RK_sol1 - train_u))
    loss_RK4 = np.mean(np.square(RK_sol4 - train_u))
    
    # Compute loss by evaluating on the given input data
    print(f"NN loss: {NN.evaluate(pred_u, train_u, verbose=0):.2E}")
    print(f"RK4 loss: {loss_RK4:.2E}")
    print(f"RK1 loss: {loss_RK1:.2E}")
    
    plt.figure(figsize = (10,8))
    plt.plot(train_t, train_u, 'ok', label = 'Train')
    plt.plot(train_t, 0.7*np.ones_like(train_t), 'co', label= 'Train*')
    plt.plot(test_t, true_u, '-k',label = 'True')
    plt.plot(np.sort(train_t), RK_sol4, '--b', label = 'RK4')
    plt.plot(np.sort(train_t), RK_sol1, linestyle = 'dotted', color = 'green', label = 'RK1')
    plt.plot(train_t, pred_u, '--r', label = 'NN')
    plt.legend(fontsize = 15)
    #plt.xticks(fontsize = 18)
    #plt.yticks(fontsize = 18)
    plt.xlabel('t', fontsize = 15)
    plt.ylabel('u', fontsize = 15)
    plt.title(f"Model comparison {itr+1}", fontsize = 15)
    #plt.show()
    plt.savefig(fname=f"Model_comparison_{itr+1}.png", bbox_inches='tight')
