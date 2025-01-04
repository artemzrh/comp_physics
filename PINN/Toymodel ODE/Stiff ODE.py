import numpy as np
import matplotlib.pyplot as plt

# Stiff ODE, i.e., RK fails
def func(y, t):
    return -15*np.exp(-15*t)

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

train_t = np.arange(0, 1+0.05, 0.05)

y0 = 1
RK_sol = rungekutta4(func, y0, np.sort(train_t))

true_u = np.exp(-15*train_t)

plt.figure(figsize = (10,8))
plt.show()

plt.figure(figsize = (10,8))
plt.plot(np.sort(train_t), RK_sol, '--b', label = 'RK4')
plt.plot(train_t, true_u, '--r', label = 'True')
plt.legend(fontsize = 15)
plt.xlabel('t', fontsize = 15)
plt.ylabel('u', fontsize = 15)
plt.show()