import numpy as np
import matplotlib.pyplot as plt

class Solve_Heat_Eq:

    def __init__(self, h, k, tf, f, t0=0, x0=0, xf=2*np.pi):
        self.h = h
        self.k = k
        self.f = f
        self.time_steps = round((tf - t0) / k)
        self.space_steps = round((xf - x0) / h)
        self.x_axis = np.linspace(x0, xf, self.space_steps)
        self.t_axis = np.linspace(t0, tf, self.time_steps)
        self.sig = k/(h**2)
        self.exact_sol = np.zeros([self.time_steps, self.space_steps])

    def Backward_Euler(self, t0):
        #To allow for the matrix to be invertible we must make it square
        self.time_steps = self.space_steps
        tf = self.time_steps * self.k 

        #We also have to recompute the exact solution grid with the new dimensions
        self.t_axis = np.linspace(t0, tf, self.time_steps)
        self.exact_sol = np.zeros([self.time_steps, self.space_steps])
        self.exact()

        #Initializing grid and operations matrix
        op_mat = np.zeros([self.time_steps, self.space_steps])
        v = np.zeros([self.time_steps, self.space_steps])
        v[0, :] = self.exact_sol[0, :]

        #Setting up left side operations matrix
        for n in range(self.time_steps):
            if n == 0:
                op_mat[n, 0] = 1 + 2*self.sig
                op_mat[n, 1] = -self.sig
                op_mat[n, self.space_steps - 1] = -self.sig
            elif n == self.time_steps - 1:
                op_mat[n, n - 1] = -self.sig
                op_mat[n, n] = 1 + 2*self.sig
                op_mat[n, 0] = -self.sig
            else:
                op_mat[n, n - 1] = -self.sig
                op_mat[n, n] = 1 + 2*self.sig
                op_mat[n, n + 1] = -self.sig

        op_mat = np.linalg.inv(op_mat)
        for n in range(self.time_steps - 1):
            v[n+1, :] = np.matmul(op_mat, v[n, :])
        
        return v

    def Crank_Nicolson(self, t0):
        self.time_steps = self.space_steps
        tf = self.time_steps * self.k 

        self.t_axis = np.linspace(t0, tf, self.time_steps)
        self.exact_sol = np.zeros([self.time_steps, self.space_steps])
        self.exact()

        l_op_mat = np.zeros([self.time_steps, self.space_steps])
        r_op_mat = np.zeros([self.time_steps, self.space_steps])
        v = np.zeros([self.time_steps, self.space_steps])
        v[0, :] = self.exact_sol[0, :]

        #Setting up left side operations matrix
        for n in range(self.time_steps):
            if n == 0:
                l_op_mat[n, 0] = 1 + self.sig
                l_op_mat[n, 1] = -self.sig/2
                l_op_mat[n, self.space_steps - 1] = -self.sig/2
            elif n == self.time_steps - 1:
                l_op_mat[n, n - 1] = -self.sig/2
                l_op_mat[n, n] = 1 + self.sig
                l_op_mat[n, 0] = -self.sig/2
            else:
                l_op_mat[n, n - 1] = -self.sig/2
                l_op_mat[n, n] = 1 + self.sig
                l_op_mat[n, n + 1] = -self.sig/2

        #Setting up right side operations matrix
        for n in range(self.time_steps):
            if n == 0:
                r_op_mat[n, 0] = 1 - self.sig
                r_op_mat[n, 1] = self.sig/2
                r_op_mat[n, self.space_steps - 1] = self.sig/2
            elif n == self.time_steps - 1:
                r_op_mat[n, n - 1] = self.sig/2
                r_op_mat[n, n] = 1 - self.sig
                r_op_mat[n, 0] = self.sig/2
            else:
                r_op_mat[n, n - 1] = self.sig/2
                r_op_mat[n, n] = 1 + self.sig
                r_op_mat[n, n + 1] = self.sig/2

        l_op_mat = np.linalg.inv(l_op_mat)
        for n in range(self.time_steps - 1):
            v[n+1, :] = np.matmul(np.matmul(l_op_mat, r_op_mat), v[n, :])

        return v
    
    def Forward_Euler(self):
        v = np.zeros([self.time_steps, self.space_steps])
        v[0, :] = self.exact_sol[0, :]
        for n in range(1, self.time_steps - 1):
            #Left and right end points
            v[n+1, 0] = v[n,0] + self.sig*(v[n, 1] - 2*v[n, 0] + v[n, self.space_steps - 1])
            v[n+1, self.space_steps - 1] = v[n+1, 0]


            for j in range(1, self.space_steps - 1):
                v[n+1, j] = v[n, j] + self.sig*(v[n, j+1] - 2*v[n, j] + v[n, j-1])
            
        return v

    def Dufort_Frankel(self):
        v = np.zeros([self.time_steps, self.space_steps])
        v[0, :] = self.exact_sol[0, :]
        for n in range(1, self.time_steps - 1):
            #Left and right end points
            nmin1_coeff = (1 - 2*self.sig) / (1 + 2 * self.sig)
            ncoeff = (2*self.sig) / (1 + 2 * self.sig)

            v[n+1, 0] = nmin1_coeff*v[n - 1, 0] + ncoeff*(v[n, 1] + v[n, self.space_steps - 1])
            v[n+1, self.space_steps - 1] = v[n+1, 0]


            for j in range(1, self.space_steps - 1):
                v[n+1, j] = nmin1_coeff*v[n - 1, j] + ncoeff*(v[n, j+1] + v[n, j-1])
            
        return v

