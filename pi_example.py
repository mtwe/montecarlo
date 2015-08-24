import numpy as np
import montecarlo
import matplotlib.pyplot as plt

def calculate_pi():
    """
    Function: calculate_pi
    Summary: A generator to estimate the value of pi by 'throwing darts' at a circle
    Examples: See http://mathfaculty.fullerton.edu/mathews/n2003/MonteCarloPiMod.html for further explanation
    Yields: a single 'dart's' estimation of pi, either 0 or 4 dependeing if it landed outside or inside the circle
    """
    while True:
        xi_1 = np.random.random_sample()
        x = 2*xi_1 - 1
        y = 2*np.random.random_sample() - 1
        if x**2 + y**2 < 1.0:
            yield 4.0
        else:
            yield 0.0

pi_ex1 = montecarlo.MonteCarlo(calculate_pi(), degree=6) #Monte Carlo experiment that scores an estimation of pi 10^6 times
pi_ex1.run()
pi_ex1.print_results()
ax,probs,bins = pi_ex1.plot_histogram()

ax.set_title('Estimate of Pi')
ax.set_ylabel('Normalized Probability')

plt.show()