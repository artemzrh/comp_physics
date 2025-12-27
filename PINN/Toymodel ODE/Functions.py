import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def plotting_points(epochs):
    
    points = np.linspace(0, epochs, 10, dtype=int, endpoint=True)
    
    if 0 <= epochs <= 101:
        points = [0, 20, 40, 80]
    elif 101 < epochs <= 201:
        points = [0, 50, 100, 150]
    elif 201 < epochs <= 501:
        points = [0, 75, 150, 250, 350, 450]
    elif 501 < epochs <= 1001:
        points = [0, 100, 250, 450, 650, 850]
    elif 1001 < epochs <= 2001:
        points = [0, 250, 500, 750, 1000, 1250, 1500, 1750]
    elif 2001 < epochs <= 5001:
        points = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    
    return points

# Generating impulse noise
def impulse_noise(true_u):
    
    # Generate a noise sample consisting of values that are a little higer or lower than a few randomly selected values in the original data
    noise_sample = np.random.default_rng().uniform(0.05*min(true_u), 0.1*max(true_u), int(0.1*len(true_u)))
    #noise_sample = np.random.default_rng().uniform(0.05*np.ma.masked_invalid(true_u).min(), 0.1*np.ma.masked_invalid(true_u).max(), int(0.2*len(true_u)))

    # Generate an array of zeros with a size that is the difference of the sizes of the original data an the noise sample
    zeros = np.zeros(len(true_u) - len(noise_sample))

    # Add the noise sample to the zeros array to obtain the final noise with the same shape as that of the original data
    noise = np.concatenate([noise_sample, zeros])

    # Shuffle the values in the noise to make sure the values are randomly placed
    np.random.shuffle(noise)

    # Obtain data with the noise added
    true_u_noised = true_u + noise
    
    return true_u_noised

# Euler method
def rungekutta1(f, y0, t, args=()):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        y[i+1] = y[i] + (t[i+1] - t[i]) * f(y[i], t[i], *args)
    return y

# RK4: fourth-order iterative method
# taken from https://perso.crans.org/besson/publis/notebooks/Runge-Kutta_methods_for_ODE_integration_in_Python.html
def rungekutta4(f, y0, t, args=()):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = f(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y

def rk4_error(u_pred, u_true):
    return np.abs(u_pred-u_true)

# Initialize Neural Network
def create_NN_model():
    neural_net = tf.keras.models.Sequential([
        tf.keras.layers.Input((1,)),
        tf.keras.layers.Dense(units = 32, activation = 'tanh'),
        tf.keras.layers.Dense(units = 32, activation = 'tanh'),
        tf.keras.layers.Dense(units = 32, activation = 'tanh'),
        tf.keras.layers.Dense(units = 1)
    ])
    return neural_net

# Define ODE system with physical loss function
def ode_system(t, net, func, bc):
    t = t.reshape(-1,1)
    t = tf.constant(t, dtype = tf.float32)
    t_0 = tf.zeros((1,1))
    one = bc

    with tf.GradientTape() as tape:
        tape.watch(t)

        u = net(t)
        u_t = tape.gradient(u, t)

    ode_loss = u_t - func(t)
    IC_loss = net(t_0) - one

    #square_loss = tf.square(ode_loss) + tf.square(IC_loss)
    #total_loss = tf.reduce_mean(square_loss)
    
    total_loss = tf.reduce_mean(tf.square(ode_loss)) + tf.reduce_mean(tf.square(IC_loss))

    return total_loss

# Custom training routine
def train_NN(epochs, optm, NN, func, bc, train_t, train_u, test_t, true_u):
    train_loss_record = []
    #loss_tracker = np.linspace(0, epochs, 10, dtype=int, endpoint=True)
    loss_tracker = plotting_points(epochs)
    
    patience = 200
    wait = 0
    best = float('inf') 
    
    early_stop = 0
    
    for itr in range(epochs):
        with tf.GradientTape() as tape:
            train_loss = ode_system(train_t, NN, func, bc)
            train_loss_record.append(train_loss)

            grad_w = tape.gradient(train_loss, NN.trainable_variables)
            optm.apply_gradients(zip(grad_w, NN.trainable_variables))

        if itr in loss_tracker:
            print(train_loss.numpy())
            plot_epochs(train_t, train_u, test_t, true_u, itr, NN)
            
        # early stopping
        wait += 1
        if train_loss.numpy() < best:
            best = train_loss.numpy()
            wait = 0
        if wait >= patience:
            print(f"Stopped at iteration {itr} with loss {train_loss_record[itr]}.")
            early_stop = itr
            break
            
    return train_loss_record, early_stop

# Including both, physical and data loss, for ODE system
def ode_system_data_loss(t, net, func, bc, t_data, u_data, lambda1):
    t = t.reshape(-1,1)
    t = tf.constant(t, dtype = tf.float32)
    
    t_data = t_data.reshape(-1,1)
    t_data = tf.constant(t_data, dtype = tf.float32)
    
    t_0 = tf.zeros((1,1))
    one = bc

    with tf.GradientTape() as tape:
        tape.watch(t)

        u = net(t)
        u_t = tape.gradient(u, t)

    ode_loss = u_t - func(t)
    IC_loss = net(t_0) - one
    
    # data loss with hyperparameter lambda1
    #data_los = u_data - net(t_0)
    # Need to have the t values of the noisy data **********************
    #data_los = u_data - net(t)
    data_loss = u_data - net(t_data)

    #square_loss = tf.square(ode_loss) + tf.square(IC_loss) + lambda1*tf.square(data_loss)
    #total_loss = tf.reduce_mean(square_loss)
    
    total_loss = tf.reduce_mean(tf.square(ode_loss)) + tf.reduce_mean(tf.square(IC_loss)) + lambda1*tf.reduce_mean(tf.square(data_loss))

    return total_loss

# Training routine with data loss
def train_NN_data_loss(epochs, optm, NN, func, bc, lambda1, train_t, train_u, data_t, data_u,
                       data_u_noised, test_t_plot, true_u_plot, testing_t):
    train_loss_record = []
    #loss_tracker = np.linspace(0, epochs, 10, dtype=int, endpoint=True)
    # epsilon = 0.00001
    loss_tracker = plotting_points(epochs)
    
    patience = 70
    wait = 0
    best = float('inf')
    
    early_stop = 0
    
    for itr in range(epochs):
        with tf.GradientTape() as tape:
            train_loss = ode_system_data_loss(train_t, NN, func, bc, data_t, data_u_noised, lambda1)
            train_loss_record.append(train_loss)
    
            grad_w = tape.gradient(train_loss, NN.trainable_variables)
            optm.apply_gradients(zip(grad_w, NN.trainable_variables))
    
        if itr in loss_tracker:
            print(train_loss.numpy())
            plot_epochs_with_noise(train_t, train_u, data_t, data_u_noised, test_t_plot, true_u_plot, testing_t, itr, NN)
    
        # early stopping
        # wait += 1
        # if train_loss.numpy() < best:
        #     best = train_loss.numpy()
        #     wait = 0
        # if wait >= patience:
        #     print(f"Stopped at iteration {itr} with loss {train_loss_record[itr]}.")
        #     early_stop = itr
        #     break
            
    return train_loss_record, early_stop

# Plot intermediate training steps
def plot_epochs(train_t, train_u, test_t, true_u, itr, NN, name = "undefined"):
    
    pred_u = NN.predict(test_t).ravel()
    
    plt.figure(figsize = (10,8))
    plt.plot(train_t, train_u, 'ok', label = 'Train')
    plt.plot(test_t, true_u, '-k',label = 'True')
    plt.plot(test_t, pred_u, '--r', label = 'NN')
    plt.legend(fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.xlabel('t', fontsize = 20)
    plt.ylabel('u', fontsize = 20)
    plt.title(f"Training step {itr}", fontsize = 20)
    if name == "undefined":
        plt.show()
    else:
        plt.savefig(name, bbox_inches='tight')

def plot_epochs_with_noise(train_t, train_u, data_t, data_u_noised, test_t_plot, true_u_plot, testing_t, itr, NN, a=0, b=0, stiff=False, path="undefined"):
    
    #a_var = 4.444
    #b_var = -0.112
    #pred_u = tf.cast(NN(testing_t), tf.float64)*tf.sin(a_var*testing_t+b_var)[:, tf.newaxis]
    
    #pred_u = tf.cast(NN(testing_t), tf.float64)*tf.sin(a*testing_t+b)[:, tf.newaxis]    # with ansatz formulation
    pred_u = NN.predict(testing_t)
    
    plt.figure(figsize = (10,8))
    plt.plot(train_t, train_u, 'ok', label = 'Train')
    if stiff == False:
        plt.plot(data_t, data_u_noised, 'o',label = 'Noise', color='gray')
    plt.plot(test_t_plot, true_u_plot, '-k',label = 'Exact')
    plt.plot(testing_t, pred_u, '--r', label = 'NN')
    plt.legend(fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.xlabel('t', fontsize = 20)
    plt.ylabel('u', fontsize = 20)
    plt.title(f"Training step {itr}", fontsize = 15)
    if path == "undefined":
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')

# Plot final fit
def plot_figs(train_t, train_u, test_t, true_u, RK_sol, pred_u, epochs = 0, name="undefined"):
    
    plt.figure(figsize = (10,8))
    plt.plot(train_t, train_u, 'ok', label = 'Train')
    plt.plot(test_t, true_u, '-k',label = 'True')
    plt.plot(np.sort(train_t), RK_sol, '--b', label = 'RK4')
    plt.plot(test_t, pred_u, '--r', label = 'NN')
    plt.legend(fontsize = 15)
    plt.xlabel('t', fontsize = 15)
    plt.ylabel('u', fontsize = 15)
    plt.title(f"Model fit, {epochs-1} epochs", fontsize = 15)
    if name == "undefined":
        plt.show()
    else:
        plt.savefig(name, bbox_inches='tight')
    
def plot_figs_with_noise(train_t, train_u, data_t, data_u_noised, test_t_plot, true_u_plot, RK_sol, testing_t, pred_u, epochs = 0, stiff = False):
    
    plt.figure(figsize = (10,8))
    plt.plot(train_t, train_u, 'ok', label = 'Train')
    if stiff == False:
        plt.plot(data_t, data_u_noised, 'o',label = 'Noise', color='gray')
    plt.plot(test_t_plot, true_u_plot, '-k',label = 'True')
    plt.plot(testing_t, pred_u, '--r', label = 'NN')
    plt.legend(fontsize = 15)
    plt.xlabel('t', fontsize = 15)
    plt.ylabel('u', fontsize = 15)
    plt.title(f"Model fit, {epochs} epochs", fontsize = 15)
    plt.show()
    
# Plot model loss
def plot_loss(train_loss_record, path="undefined"):
    
    plt.figure(figsize = (10,8))
    plt.plot(train_loss_record, color='r')
    #plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits=(0,-3))
    plt.xlabel('Epoch', fontsize = 15)
    plt.ylabel('Loss', fontsize = 15)
    plt.ylim(ymin=0)
    plt.grid()
    plt.title("Model loss", fontsize = 15)
    if path == "undefined":
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')