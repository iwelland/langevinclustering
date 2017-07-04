import numpy as np
cimport numpy as np
import random
#from mpi4py import MPI
#from mpi4py cimport MPI

#ctypedef np.float64 float64

def potential_old(np.ndarray x,np.ndarray data, double sigma):

    cdef double term2
    cdef double potential = 0
    cdef np.ndarray prodx = x*x
    cdef np.ndarray xnorm = prodx.sum(1)
    cdef np.ndarray proddata = data*data
    cdef np.ndarray datanorm = proddata.sum(1)
    
    for i in xrange(len(x)):
        term= (xnorm[i] - 2*np.dot(x[i],data[i]) + datanorm[i])*(2*sigma**2)**-1
        potential +=term
        #term2 = np.dot(diff,diff)/(2*sigma**2)
    return potential
        
def potential_constant(np.ndarray data, double sigma):

    cdef np.ndarray datanorm = data*data
    cdef pc = np.exp(-datanorm).sum()
    
    return pc

def potential(np.ndarray x,np.ndarray data, double sigma, double pc):

    cdef double term2
    cdef double potential = pc
    cdef np.ndarray prodx = x*x
    cdef np.ndarray xnorm = prodx.sum(1)
    cdef np.ndarray xdotdata = np.dot(x,data.T)
    
   

    for i in xrange(len(x)):
        term1 = xnorm[i]/(2*sigma**2)
        term2 = 1/(2*sigma**2)*np.log(np.exp(-2*xdotdata[i]).sum())
        potential += term1 + term2
        #term2 = np.dot(diff,diff)/(2*sigma**2)
    return potential

def dynamics(x,t):
    pass

    
def monte_carlo(np.ndarray x,np.ndarray data,double pc,double sigma=1,double scale=1,int t=100,int stride=10,double temperature=100):

    cdef int x_size = x.shape[0]
    cdef int y_size = x.shape[1]
    cdef np.ndarray trajectory = np.empty((t/stride,len(x),len(x[0])),dtype=np.float64)
    cdef np.ndarray x_0 = np.empty((x_size,y_size),dtype=np.float64)
    cdef np.ndarray x_trial = np.empty((x_size,y_size),dtype=np.float64)
    cdef double p0
    cdef double pt
    cdef np.ndarray x_new = np.empty((x_size,y_size),dtype=np.float64)
    cdef double index
    cdef int accepted
    cdef np.ndarray random_gauss
    cdef double ep0
    cdef double ept

    accepted = 0
    trajectory = np.empty((t/stride,len(x),len(x[0])))
    trajectory[0] = data[0]
    for step in xrange(t):
        if step == 0:
            x_0 = x
        x_trial = np.random.normal(x_0,scale=scale)
        p0 = potential(x_0,data,sigma,pc)
        pt = potential(x_trial,data,sigma,pc)
        ep0 = np.exp(-p0*(temperature)**-1)
        ept = np.exp(-pt*(temperature)**-1)
        if pt < p0:
            accepted +=1
            x_0 = x_trial
        elif pt >= p0:
            alpha = np.random.uniform()
            if alpha < ept/(ep0+ept):
                x_0 = x_trial
                accepted+=1
            else:
                x_new = x_0
        if step % stride == 0:
            trajectory[step/stride]=x_0
    print(float(accepted)/float(t))
    return trajectory


def single_particle_random_monte_carlo(np.ndarray x,np.ndarray data,double sigma,double scale=1,int t=100,int stride=10):

    cdef int x_size = x.shape[0]
    cdef int y_size = x.shape[1]
    cdef np.ndarray trajectory = np.empty((t/stride,len(x),len(x[0])),dtype=np.float64)
    cdef np.ndarray x_0 = np.empty((x_size,y_size),dtype=np.float64)
    cdef np.ndarray x_trial = np.empty((x_size,y_size),dtype=np.float64)
    cdef double p0
    cdef double pt
    cdef np.ndarray x_new = np.empty((x_size,y_size),dtype=np.float64)
    cdef int index
    trajectory[0] = x
    for step in xrange(t):
        if step == 0:
            x_0 = x
        else:
            x_0 = trajectory[step/stride-1]
        index = np.random.choice(len(data))
        #x_trial = np.empty(x_0.shape)
        x_trial[::] = x_0
        x_trial[index] = np.random.normal(x_0[index],scale=scale)
        p0 = np.exp(-potential(x_0,data,sigma)*scale**-1)
        pt = np.exp(-potential(x_trial,data,sigma)*scale**-1)
        if pt < p0:
            x_new = x_trial
        elif pt > p0:
            alpha = np.random.uniform()
            if alpha < pt/(pt + p0):
                x_new = x_trial
            else:
                x_new = x_0
        if step % stride == 0:
            trajectory[step/stride]=x_new
    return trajectory

def single_particle_sequential_monte_carlo(np.ndarray x,np.ndarray data,double sigma,double scale=1,int t=100,int stride=10):
    cdef int x_size = x.shape[0]
    cdef int y_size = x.shape[1]
    cdef np.ndarray trajectory = np.empty((t/stride,len(x),len(x[0])),dtype=np.float64)
    cdef np.ndarray x_0 = np.empty((x_size,y_size),dtype=np.float64)
    cdef np.ndarray x_trial = np.empty((x_size,y_size),dtype=np.float64)
    cdef double p0
    cdef double pt
    cdef np.ndarray x_new = np.empty((x_size,y_size),dtype=np.float64)
    for step in xrange(t):
        if step == 0:
            x_0 = x.astype(np.float64)
        else:
            x_0 = trajectory[step/stride-1]
        
        #x_trial = np.empty(x_0.shape)
        x_trial[::] = x_0
        for particle in xrange(len(x_trial)):
            x_trial[particle] += np.random.normal(x_0[particle],scale=scale)
            p0 = np.exp(-potential(x_0,data,sigma)*scale**-1)
            pt = np.exp(-potential(x_trial,data,sigma)*scale**-1)
            if pt < p0:
                x_new = x_trial
            elif pt > p0:
                alpha = np.random.uniform()
                if alpha < p0/pt:
                    x_0 = x_new
                    x_new = x_trial
                    
                else:
                    x_new = x_0
        if step % stride == 0:
            trajectory[step/stride]=x_new
    return trajectory

def MD(x,data,sigma,time=100,stride=10):
    pass

def propagator(x,data,sigma,time=100,stride=10):
    pass
    


    


    
