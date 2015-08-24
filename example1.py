import numpy as np
import montecarlo
import matplotlib.pyplot as plt

def random_num():
    """
    Function: random_num
    Summary: A generator implementation of numpy.ranoom.random_sample, yields a random number in [0.0,1.0)
    """
    while True:
        yield np.random.random_sample()

ex1 = montecarlo.MonteCarlo(random_num(), degree=4) #Monte Carlo experiment that scores a random number 10^4 times
ex1.run()
ex1.print_results()
ax,probs,bins = ex1.plot_histogram()
ax.set_title('Distribution of sampled random numbers in [0.0,1.0)')
ax.set_ylabel('Normalized probability')
plt.show()
