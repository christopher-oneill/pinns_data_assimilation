import numpy as np
import matplotlib.pyplot as plot

m = 1
c = 0.1
k = (np.power(np.pi,2))

dt =0.001
nt = 100000

t = np.linspace(0,(nt-1)*dt,nt)
x = np.zeros(nt)
x[0]=1
x_dot = np.zeros(nt)
x_dot_dot = np.zeros(nt)

for i in range(1,x.size):
    x_dot_dot[i] = -(c/m)*x_dot[i-1]-(k/m)*x[i-1]
    x_dot[i]=x_dot[i-1]+dt*x_dot_dot[i]
    x[i] = x[i-1]+dt*x_dot[i]

plot.plot(t,x)
plot.show()




