import numpy as np

def random_num():
    """
    Function: random_num
    Summary: A generator implementation of numpy.ranoom.random_sample, yields a random number in [0.0,1.0)
    """
    while True:
        yield np.random.random_sample()

def sample_uniform(domain):
    """samples randomly from a uniform distribution with the x endpoints
    as the two elements of the domain tuple"""
    a = min(domain)
    b = max(domain)
    rn = random_num()
    while True:
        yield a + next(rn)*(b - a)

def sample_uniform2(domain):
    '''
    Function: sample_uniform2
    Summary: function that randomly samples, not a generator
    Examples: InsertHere
    Attributes: 
        @param (domain):tuple of domain over which to sample
    Returns: sampled value in domain
    '''
    return min(domain) + np.random.random_sample()*(max(domain) - min(domain))


def discrete_pdf2cdf(probs):
    '''
    Function: discrete_pdf2cdf
    Summary: Takes a 1D list of probs and converts to cdf list of same size
    Examples: [0.1,0.2,0.5,0.3] -> [0.1,0.3,0.8,1.0]
    Attributes: 
        @param (probs):list of probs. Need not sum to 1 but cannot exceed 1
    Returns: list
    '''
    assert(sum(probs) <= 1)
    cdf = [i for i in probs]
    for i in range(1,len(cdf)):
        cdf[i] += cdf[i-1]
    for i in range(len(cdf)):
        cdf[i] /= cdf[-1]
    return cdf

def sample_cdf(cdf):
    '''
    Function: sample_cdf
    Summary: Samples from a cdf using a rng
    Examples: InsertHere
    Attributes: 
        @param (cdf):list of values that add up to 1
    Returns: index of sampled value
    '''
    assert(cdf[-1]==1)
    rn = np.random.random_sample()
    for i,p in enumerate(cdf):
        if rn <= p:
            return i

def rejection(pdf,domain):
    """MonteCarlo rejection method to select a random number from PDF
    First samples from a uniform probability distribution over domain input 
    and then keeps the sample with probability proportional to pdf input
    
    NOTE: PDF need NOT be normalized
    """
    a = min(domain)
    b = max(domain)
    uniform = sample_uniform(domain)
    #calculate fsup
    fmax = max([pdf(x) for x in np.linspace(a,b,100)])
    fsup = 1.1*fmax
    rn = random_num()
    while True:
        #get random number x1 from domain uniform dist
        x1 = next(uniform)
        #get random number x2 from fsup uniform dist
        x2 = next(rn)*fsup
        #compare x2 to pdf(x1)
        if x2 < pdf(x1):
            yield x1
        else:
            pass

