import numpy as np
import tensorflow as tf
#import random
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

def RK_plots(t_RK, RK_sol, regime):
    
    # plt.figure(figsize = (10,8))
    # plt.plot(t, RK_sol[:, 0], label='with 1001 points')
    # plt.legend(loc='best')
    # plt.xlabel('t', fontsize = 15)
    # plt.ylabel('u', fontsize = 15)
    # plt.grid()
    # plt.title("RK4 fit", fontsize = 15)
    # plt.show()
    
    plt.figure(figsize = (10,8))
    plt.plot(t_RK, RK_sol[:, 0], 'b', label=r'$x(t)$')
    plt.plot(t_RK, RK_sol[:, 1], 'g', label=r'$v(t)$')
    plt.legend(loc='best')
    plt.xlabel('t', fontsize = 15)
    plt.ylabel('u', fontsize = 15)
    plt.grid()
    plt.title(f"Runge-Kutta 4, {regime}", fontsize = 15)
    plt.show()

def plot_figs_with_noise(train_t, train_u, data_t, data_u_noised, test_t_plot, true_u_plot, RK_sol, testing_t, pred_u, regime, epochs = 0, path="undefined"):
    
    plt.figure(figsize = (10,8))
    plt.plot(train_t, train_u, 'ok', label = 'Train')
    plt.plot(data_t, data_u_noised, 'o',label = 'Noise', color='gray')
    plt.plot(test_t_plot, true_u_plot, '-k',label = 'True')
    plt.plot(testing_t, pred_u, '--r', label = 'NN')
    plt.legend(fontsize = 15)
    plt.xlabel('t', fontsize = 15)
    plt.ylabel('u', fontsize = 15)
    plt.title(f"Model fit, {epochs} epochs, {regime}", fontsize = 15)
    if path == "undefined":
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
    
def plot_epochs_with_noise(train_t, train_u, data_t, data_u_noised, test_t_plot, true_u_plot, testing_t, itr, NN, regime, path="undefined"):
    
    pred_u = NN.predict(testing_t)
    
    plt.figure(figsize = (10,8))
    plt.plot(train_t, train_u, 'ok', label = 'Train')
    plt.plot(data_t, data_u_noised, 'o',label = 'Noise', color='gray')
    plt.plot(test_t_plot, true_u_plot, '-k',label = 'Exact')
    plt.plot(testing_t, pred_u, '--r', label = 'NN')
    plt.legend(fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.xlabel('t', fontsize = 20)
    plt.ylabel('u', fontsize = 20)
    plt.title(f"Training step {itr}, {regime}", fontsize = 20)
    if path == "undefined":
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')

# Plots of bc-optimization
def plot_bc_optim(bc1_list, bc2_list, regime, path="undefined"):
    
    plt.figure(figsize = (10,8))
    plt.plot(bc1_list, color='r')
    plt.xlabel('Epoch', fontsize = 20)
    plt.ylabel(r'$x_0$', fontsize = 20)
    plt.grid()
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.title(f"Initial position, {regime}", fontsize = 20)
    if path == "undefined":
        plt.show()
    else:
        plt.savefig(f"PINN_{regime}_position.png", bbox_inches='tight')

    plt.figure(figsize = (10,8))
    plt.plot(bc2_list, color='r')
    plt.xlabel('Epoch', fontsize = 20)
    plt.ylabel(r'$v_0$', fontsize = 20)
    plt.grid()
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.title(f"Initial velocity, {regime}", fontsize = 20)
    if path == "undefined":
        plt.show()
    else:
        plt.savefig(f"PINN_{regime}_velocity.png", bbox_inches='tight')

def plot_loss(train_loss_record, regime, path="undefined"):
    
    plt.figure(figsize = (10,8))
    plt.plot(train_loss_record, color='r')
    plt.xlabel('Epoch', fontsize = 15)
    plt.ylabel('Loss', fontsize = 15)
    plt.ylim(ymin=0)
    plt.grid()
    plt.title(f"Model loss, {regime}", fontsize = 15)
    if path == "undefined":
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')

# NN architecture
def create_NN_model():
    neural_net = tf.keras.models.Sequential([
        tf.keras.layers.Input((1,)),
        tf.keras.layers.Dense(units = 32, activation = 'tanh'),
        tf.keras.layers.Dense(units = 32, activation = 'tanh'),
        tf.keras.layers.Dense(units = 32, activation = 'tanh'),
        tf.keras.layers.Dense(units = 1)
    ])
    return neural_net

# Complete analytical solutions harmonic oscillator problem
def exact_solution(t, delta, w0, A0, v0):
    # underdamped
    if delta < w0:
        w = np.sqrt(w0**2-delta**2)
        phi = np.arctan(1/w*(v0/A0-delta))
        A = A0/(np.cos(phi))
        u = np.exp(-delta*t)*A*np.cos(phi+w*t)
        return u
    # overdamped
    elif delta > w0:
        w = np.sqrt(delta**2-w0**2)
        x1 = 1/2*(1/w*(v0+A0*delta)+A0)
        x2 = A0-x1
        u = np.exp(-delta*t)*(x1*np.exp(w*t)+x2*np.exp(-w*t))
        return u
    # critical
    elif delta == w0:
        x1 = A0
        x2 = v0 + delta*A0
        u = np.exp(-delta*t)*(x1+x2*t)
        return u

# Implementation of oscillator system for NN
def oscillator_system_data_loss(t, net, func, params, bc, t_data, u_data, lambda1):
    t = t.reshape(-1,1)
    t = tf.constant(t, dtype = tf.float32)
    t_0 = tf.zeros((1,1))
    
    # Nested loop for 2nd derivative
    with tf.GradientTape() as outer_tape:
        outer_tape.watch(t)
         
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(t)
            x = net(t)
     
        dx_dt = inner_tape.gradient(x, t)  # 1st derivative
 
    d2x_dt2 = outer_tape.gradient(dx_dt, t)  # 2nd derivative
    
    # Boundary loss
    bc_loss_1 = tf.square(net(t_0) - bc[0])
    bc_loss_2 = tf.square(dx_dt[0] - bc[1])
        
    ode_loss = d2x_dt2 - func(x, dx_dt, params[0], params[1], params[2])
    
    # Data loss with hyperparameter lambda1
    data_loss = u_data - net(t_data)
    
    total_loss = tf.reduce_mean(tf.square(ode_loss)) + lambda1*tf.reduce_mean(tf.square(data_loss)) + tf.reduce_mean(bc_loss_1) + tf.reduce_mean(bc_loss_2)

    return total_loss, net(t_0), dx_dt[0]

# Training routine with data loss
def train_NN_data_loss(epochs, optm, NN, func, bc, lambda1, train_t, train_u, data_t, data_u,
                       data_u_noised, test_t_plot, true_u_plot, testing_t):
    
    train_loss_record = []
    bc1_list = []
    bc2_list = []
    
    early_stop = 0
    
    for itr in range(epochs+1):
        with tf.GradientTape() as tape:
            train_loss, bc1, bc2 = oscillator_system_data_loss(train_t, NN, func, params, bc, data_t, data_u_noised, lambda1)
            train_loss_record.append(train_loss)
            bc1_list.append(bc1.numpy()[0][0])
            bc2_list.append(bc2.numpy()[0])
    
            grad_w = tape.gradient(train_loss, NN.trainable_variables)
            optm.apply_gradients(zip(grad_w, NN.trainable_variables))
    
        if (itr % 500) == 0:
            print(train_loss.numpy(), bc1.numpy()[0][0], bc2.numpy()[0])
            plot_epochs_with_noise(train_t, train_u, data_t, data_u_noised, test_t_plot, true_u_plot, testing_t, itr, NN, regime, f"PINN_{regime}_training_{itr}.png")
            
    return train_loss_record, early_stop, bc1_list, bc2_list

# For ODE loss computation / minimization
NN_osc_func = lambda x, dx_dt, k, d, m: -k/m*x - d/m*dx_dt

# 2nd order ODE converted into 1st order ODE system
def oscillator(y, t, k, b, m):
    return np.array([y[1], -k/m*y[0] - b/m*y[1]])

# RK 4th order for converted 2nd order ODE
def rungekutta4(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = f(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y

########################################################################

endpoint = 10
m = 1
y0 = np.array([2.4, 0.0])
bc = [tf.ones((1,1))*2.4, tf.zeros((1,1))]

epochs = 3500 #3300

optm = tf.keras.optimizers.Adam(learning_rate = 0.001)

# Hyperparameter for data loss weight
# Benchmark error of 0.005963283 for lambda1 of 0.001 and 3000 epochs
# Benchmark error of 0.0015944085 for lambda1 of 0.0005 and 3000 epochs
lambda1 = 0.0005

t_RK = np.linspace(0, endpoint, 1000, endpoint=True)

# Training data
train_t = np.linspace(0, endpoint, 100, endpoint=True) #np.arange(0, endpoint+0.1, 0.1)

# True result for plotting
test_t_plot = np.linspace(0, endpoint, 100, endpoint=True)

# Data for adding noise
data_t = np.sort(10 * np.random.random_sample((len(train_t),)))

# Testing data points for NN prediction
testing_t = np.linspace(0, endpoint, 1000, endpoint=True)

regime_list = ["underdamped", "overdamped", "critical"] #"underdamped", "overdamped", 

final_loss = []

for regime in regime_list:
    
    tf.keras.backend.clear_session()
    
    if regime == "underdamped":
        k = 2; d = 0.2; epochs = 3500
    elif regime == "overdamped":
        k = 2; d = 10; epochs = 1000
    elif regime == "critical":
        k = 4; d = 4; epochs = 1000
    
    params = [k, d, m]
        
    RK_sol = rungekutta4(oscillator, y0, t_RK, args=(k, d, m))
    RK_plots(t_RK, RK_sol, regime)
    
    delta = d/(2*m)
    w0 = np.sqrt(k/m)
    
    train_u = exact_solution(train_t, delta, w0, y0[0], y0[1]) 
    true_u_plot = exact_solution(test_t_plot, delta, w0, y0[0], y0[1])
    data_u = exact_solution(data_t, delta, w0, y0[0], y0[1])
    
    # Introduce noise to observations
    data_u_noised = data_u + np.random.normal(loc=0.0, scale=0.07, size=data_t.shape)

    NN = create_NN_model()
    
    # Train NN
    train_loss_record, early_break, bc1_list, bc2_list = train_NN_data_loss(epochs, optm, NN, NN_osc_func,
                                                            bc, lambda1, train_t, train_u, data_t, data_u,
                                                            data_u_noised, test_t_plot, true_u_plot, testing_t)

    pred_u_pre_trained = NN(testing_t)

    if early_break:
        epochs = early_break

    plot_figs_with_noise(train_t, train_u, data_t, data_u_noised, test_t_plot, true_u_plot, RK_sol, testing_t, pred_u_pre_trained, regime, epochs, path=f"PINN_{regime}_fit_{epochs}.png")

    plot_loss(train_loss_record, regime, f"PINN_loss_{regime}.png")
    
    final_loss.append(train_loss_record[-1].numpy())

    plot_bc_optim(bc1_list, bc2_list, regime, path="plot")

