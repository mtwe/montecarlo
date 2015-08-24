import numpy as np
import matplotlib.pyplot as plt
import timeit

def timer_wrapper(outer):
    """
    wrapper decorator that times the decorated function
    :param outer: decorated function
    :return: decorated function
    """
    def inner(*args, **kwargs):
        start = timeit.default_timer()
        ret = outer(*args, **kwargs)
        print("Time: {:.4f} sec\n".format(timeit.default_timer()-start))
        return ret
    return inner

class MonteCarlo(object):
    """
    Class that runs a monte carlo sample and provides some output and plotting
    :param mc_sample: an instantiated generator that yields a single Monte Carlo score
    :param degree: Controls how many Monte Carlo runs to do. Will run 10^(degree) runs. 
        Ex: degree=6 implies 1 million runs"""
    def __init__(self, mc_sample, degree = 6):
        # super(MonteCarlo, self).__init__()
        self.mc_sample = mc_sample
        self.degree = degree
        # self.run()


    @property
    def degree(self):
        """
        The class runs 10^degree Monte Carlo runs
        """
        return self._degree
    @degree.setter
    def degree(self, value):
        self._degree = value
        self.num_runs = 10**self.degree
    
    @timer_wrapper
    def run(self,mc_sample=None):
        """
        Runs the Monte Carlo experiment. Optional input attribute to set a new generator for the Monte Carlo score
        """
        if mc_sample:
            self.mc_sample = mc_sample

        total_scores = 0.0
        total_scores_square = 0.0
        self.scores_list =[]
        
        for i in range(self.num_runs): #runs the specified number of Monte Carlo samples
            score = next(self.mc_sample) #next score
            self.scores_list.append(score) 
            total_scores += score
            total_scores_square += score**2

        self.xhat = total_scores / self.num_runs #mean of score
        self.x2hat = total_scores_square / self.num_runs #mean of score^2

        self.sample_variance = (self.num_runs / (self.num_runs - 1.0)) * (self.x2hat - (self.xhat**2))
        self.sample_stddev = np.sqrt(self.sample_variance)
        self.mean_variance = self.sample_variance / (self.num_runs - 1.0)
        self.mean_stddev = np.sqrt(self.mean_variance)

    def print_results(self):
        print("\033[94m"+"Summary\n"+"-"*32+"\033[0m")
        print("Subroutine: {}".format(self.mc_sample.__name__))
        print("Num Runs: {:2.1e}".format(self.num_runs))
        print("-"*32)
        print("Mean\n")
        print("estimate: {:6f}".format(self.xhat))
        print("std dev : {:6f}".format(self.mean_stddev))
        print("variance: {:6f}".format(self.mean_variance))
        print("% error : {:.2f} %".format(self.mean_stddev / self.xhat * 100)) #% error in estimate of mean approx = std deviation of mean estimate/mean estimate
        print("-"*32)
        print("Distribution\n")
        print("std dev : {:6f}".format(self.sample_stddev))
        print("variance: {:6f}".format(self.sample_variance))
        print("max : {}".format(max(self.scores_list)))
        print("min : {}".format(min(self.scores_list)))
        print()
    
    def plot_histogram(self,ax=None,**kwargs):
        """
        Plots a histogram of the score, along with the mean.
        :param ax: Matplotlib Axes object to plot on can be optionally specified
        :param **kwargs can be input to change plot design
        """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        probs,bins,patches = ax.hist(self.scores_list,normed=True,label="Sample",**kwargs)
        ax.vlines(self.xhat,*ax.get_ylim(),label='Mean',color='r')
        ax.legend()
        return ax,probs,bins

    def __call__(self):
        self.run()
        
    def __repr__(self):
        return "subroutine: {}\nNumRuns: {}".format(self.mc_sample.__name__,self.num_runs)

class MultiMonteCarlo(MonteCarlo):
    """class to run a Monte Carlo simulation where the subroutine returns a list of values
    Inherits from the single output MonteCarlo class. Child class of MonteCarlo class
    :param mc_sample: an instantiated generator that yields a list of Monte Carlo score
        Ex: each iteration of mc_sample generator yields a list of scores for different variables, [var1, var2, ...]
    :param degree: Controls how many Monte Carlo runs to do. Will run 10^(degree) runs. 
        Ex: degree=6 implies 1 million runs
    :param labels: a list of strings to identify each sampled Monte Carlo score in the list of scores
        produced by the generator mc_sample. Defaults to [0,1,2,...] 
    """
    def __init__(self, mc_sample, degree = 6, labels = []):
        super().__init__(mc_sample, degree=degree)
        
        self.labels = labels
    @timer_wrapper
    def run(self,mc_sample=None):
        """
        Runs the Monte Carlo experiment. Optional input attribute to set a new generator for the Monte Carlo score
        """
        if mc_sample:
            self.mc_sample = mc_sample

        self.score_length = self.check_format(next(self.mc_sample))
        if not len(self.labels) == self.score_length:
            if len(self.labels) > 0:
                print('Label length does not match score length\nResetting labels...')
            self.labels = [i for i in range(self.score_length)]
        total_scores = np.zeros(self.score_length)
        total_scores_square = np.zeros(self.score_length)
        self.scores_list =[]
        
        for i in range(self.num_runs):
            score = next(self.mc_sample)
            self.scores_list.append(score)
            for s in range(len(score)):
                total_scores[s] += score[s]
                total_scores_square[s] += score[s]**2

        self.xhat = total_scores / self.num_runs
        self.x2hat = total_scores_square / self.num_runs

        self.sample_variance = (self.num_runs / (self.num_runs - 1.0)) * (self.x2hat - (self.xhat**2))
        self.sample_stddev = np.sqrt(self.sample_variance)
        self.mean_variance = self.sample_variance / (self.num_runs - 1.0)
        self.mean_stddev = np.sqrt(self.mean_variance)

    @staticmethod
    def check_format(score):
        """determines the dimension of a score from a Monte Carlo run"""
        try:
            for i in score:
                pass
            length = len(score)
            return length
        except TypeError:
            return 1

    def _print_results_header(self):
        """
        prints a header for the results output
        """
        print("\033[94m"+"Summary\n"+"-"*32+"\033[0m")
        print("Subroutine: {}".format(self.mc_sample.__name__))
        print("Num Runs: {:2.1e}".format(self.num_runs))
        print("-"*32+'\n')
    
    def print_results(self,verbose=False):
        self._print_results_header()
        for i in range(self.score_length):
            self.print_single(i,verbose=verbose)

    def print_results_summary(self,remove_zeros=False,trim_end_zeros=False):
        """
        prints results of monte carlo run to screen

        The stuff about removing zeros was specific to an application that Matt had. Basically you can tell it not print out
        information for any scored variable who's mean = 0
        """
        if remove_zeros:
            if trim_end_zeros:
                raise Warning('remove_zeros = False overrides trim_end_zeros=True. Removing all values with mean=zero')
            nz_ind = np.nonzero(self.xhat)
            xhats = self.xhat[nz_ind]
            sigmas = self.mean_stddev[nz_ind]
        elif trim_end_zeros:
            xhats = np.trim_zeros(self.xhat,trim='b')
            sigmas = self.mean_stddev[np.arange(xhats.size)]
        else:
            xhats = self.xhat
            sigmas = self.mean_stddev

        self._print_results_header()
        print('{: >5} {: >8}    {: >10}  {: >4}'.format('n','mean','error','pct_error'))
        for i in range(xhats.size):
            print('{0: >5} {1: >8.4g} +/- {2: >10.4g} ({3: >4.1%})'.format(i,xhats[i],sigmas[i],sigmas[i] / xhats[i]))
    
    def print_single(self, index,verbose=False):
        
        print("Score index: {}".format(self.labels[index]))
        print("-"*32)
        print("Mean\n")
        print("estimate: {:6f}".format(self.xhat[index]))
        print("std dev : {:6f}".format(self.mean_stddev[index]))
        print("variance: {:6f}".format(self.mean_variance[index]))
        print("% error : {:.2f} %".format(self.mean_stddev[index] / self.xhat[index] * 100))
        print("-"*32)
        if verbose:
            print("Distribution\n")
            print("std dev : {:6f}".format(self.sample_stddev[index]))
            print("variance: {:6f}".format(self.sample_variance[index]))
        # print("max : {}".format(max(self.scores_list[index])))
        # print("min : {}".format(min(self.scores_list[index])))
        ind_scores = [s[index] for s in self.scores_list]
        print("max : {}".format(max(ind_scores)))
        print("min : {}".format(min(ind_scores)))
        print()
    def plot_histogram(self,**kwargs):
        """
        Plots a separate histogram for each variable scored by mc_sample routine
        returns a list of the Axes objects
        """
        axes = []
        for i in range(self.score_length):
            fig = plt.figure()
            scores = np.array([s[i] for s in self.scores_list])
            probs,bins,patches = plt.hist(scores,label="Sample {}".format(self.labels[i]), **kwargs)
            plt.vlines(self.xhat,fig.get_axes().get_ylim(),label='Mean',color='r')
            plt.legend()
            axes.append(fig.get_axes())
        return axes




