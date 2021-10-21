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

    def Newmark(self):
        cof = (self.k*self.c**2) / (2 * self.h**2)
        u = np.zeros([self.time_steps, self.space_steps])
        w = np.zeros([self.time_steps, self.space_steps])
        op_mat = np.zeros([self.space_steps, self.space_steps])

        u[0, :] = self.initialf
        w[0, :] = self.initialg
        for n in range(self.space_steps):
            if n == 0:
                op_mat[n, 1] = op_mat[n, self.space_steps - 1] = -cof
            elif n == self.space_steps - 1:
                op_mat[n, n - 1] = op_mat[n, 0] =  -cof
            else:
                op_mat[n, n - 1] = op_mat[n, n + 1] = -cof
            op_mat[n, n] = 2/self.k + 2*cof
        op_mat = np.linalg.inv(op_mat)

        for t in range(self.time_steps - 1):
            right = np.ones(self.space_steps)
            for j in range(self.space_steps):
                if j == 0:
                    r = 2*w[t, j] + cof*(u[t, self.space_steps-1] - 2*u[t,j] + u[t, j+1]) + 2/self.k*u[t,j]
                elif j == self.space_steps-1:
                    r = 2*w[t, j] + cof*(u[t, j-1] - 2*u[t,j] + u[t, 0]) + 2/self.k*u[t,j]
                else:
                    r = 2*w[t, j] + cof*(u[t, j-1] - 2*u[t,j] + u[t, j+1]) + 2/self.k*u[t,j]
                right[j] = r
            u[t+1, :] = np.matmul(op_mat, right)
            w[t+1, :] = 2/self.k *(u[t+1,:] - u[t, :]) - w[t, :]
        return u


