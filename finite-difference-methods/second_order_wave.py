import numpy as np

class Solve_Second_Order_Wave:

    def __init__(self, h, k, tf, f, g, c, t0=0, x0=0, xf=2*np.pi):
        self.h = h
        self.k = k
        self.f = f
        self.c = c
        self.time_steps = round((tf - t0) / k)
        self.space_steps = round((xf - x0) / h)
        self.x_axis = np.linspace(x0, xf, self.space_steps)
        self.t_axis = np.linspace(t0, tf, self.time_steps)
        self.initialf = np.array([f(x) for x in self.x_axis])
        self.initialg = np.array([g(x) for x in self.x_axis])
        self.exact_sol = np.zeros([self.time_steps, self.space_steps])
    
    def Leap_Frog(self):
        coef = (self.h**2 * self.c **2)/(self.h**2)
        v = np.zeros([self.time_steps, self.space_steps])
        v[0, :] = self.initialf
        print(self.initialg)
        v[1, :] = self.initialf + self.k * self.initialg
        for n in range(1, self.time_steps - 1):
            #Left and right end points
            l = self.space_steps-1
            v[n+1, 0] = coef*v[n,l] + (-2*coef+2)*v[n, 0] + coef*v[n, 1] - v[n-1, 0]
            v[n+1, l] = coef*v[n,l-1] + (-2*coef+2)*v[n, l] + coef*v[n, 0] - v[n-1, l]
            
            for j in range(1, self.space_steps - 1):
                v[n+1, j] = coef*v[n,j-1] + (-2*coef+2)*v[n, j] + coef*v[n, j+1] - v[n-1, j]
        return v
