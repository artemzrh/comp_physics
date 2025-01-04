import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

loss = [0.2758711,
 0.029157832,
 0.0029259068,
 0.0039561493,
 3.2370706e-05,
 3.7732214e-06,
 3.2198457e-07]

lambda1 = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

log_x = np.log(lambda1)
log_y = np.log(loss)

# Perform linear regression
coefficients = np.polyfit(lambda1, loss, 1)

# Extract parameters
b = coefficients[0]  # Slope
intercept = coefficients[1]  # Intercept

a = np.exp(intercept)

# Generate fitted y-values from original x-values
fitted_y = np.poly1d(np.polyfit(lambda1, loss, 1))(lambda1)

# Plot original data and fitted line
plt.figure(figsize = (10,8))
plt.scatter(lambda1, loss, color='blue', label='Loss values')
plt.plot(lambda1, fitted_y, 'r-', label=r'Fitted curve: $L( \theta) = 0.03\lambda+0.00$')
plt.xlabel(r'$\lambda$', fontsize = 15)
plt.ylabel('Loss', fontsize = 15)
plt.title(r"Loss-$\lambda$ dependence", fontsize = 15)
plt.legend(fontsize = 15)
plt.grid()
plt.show()