import numpy as np
import scipy as sp
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt

#griddding
k = 200 #number of steps per 1 second
n = 10#number of space steps

time = 10 #sec

dx = 1/n  #length of one space step
dt = 1/k  #length of one time step
alpha = 1   #speed
r = alpha*dt/dx**2 #Stable factor should be less 1/2
total_time_steps = int(time/dt) #total time steps
print("Stable factor:", r)

x = np.linspace(0, 1, n+1)

#initial condition
u = np.zeros((n+1, total_time_steps+1))
u[:, 0] = np.sin(np.pi*x) #u(x,0)= f(x)
u[0, :] = 0
u[n, :] = 0

# scheme of solving
# theta = 0 (explicit scheme), theta = 1 (implicit scheme), and theta = 1/2 (Crank-Nicolson scheme)
theta = 0

#buildin of matrix A
a = np.ones(n-1)*(1/dt+2*alpha*theta/dx**2)
b = c = np.ones(n-1)*(-alpha*theta/dx**2)
A = sp.sparse.dia_matrix(([b, a, c], [-1, 0, 1]), [n-1, n-1]).toarray()

d = np.zeros(n-1)
i = np.arange(1, n)

for t in range(0, total_time_steps):
    #Building of vector d
    d[i-1] = alpha*(1 - theta)/dx**2*u[i-1, t]\
             + (1/dt-2*alpha*(1 - theta)/dx**2)*u[i, t]\
             + alpha*(1 - theta)/dx**2*u[i+1, t]
    # solving
    u[1:n, t+1] = np.linalg.solve(A, d)
    #ploting
    plt.plot(x, u[:, t+1])
    plt.ylim([0, 1])
    plt.pause(0.00001)
    plt.clf()
