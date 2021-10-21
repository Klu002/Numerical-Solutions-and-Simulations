import numpy as np
import matplotlib.pyplot as plt

class Solve_Transport_Eq:
    
    def __init__(self, h, k, tf, f, t0=0, x0=0, xf=2*np.pi):
        
        self.h = h
        self.k = k
        self.f = f
        self.time_steps = round((tf - t0) / k)
        self.space_steps = round((xf - x0) / h)
        self.x_axis = np.linspace(x0, xf, self.space_steps)
        self.t_axis = np.linspace(t0, tf, self.time_steps)
        self.CFL = k/h
        self.initial = [f(x) for x in self.x_axis]
        self.exact_sol = np.zeros([self.time_steps, self.space_steps])
        #self.exact()

    def exact(self):
        #compute exact solution
        for n in range(self.time_steps):
            for j in range(self.space_steps):
                self.exact_sol[n,j] = self.f(self.t_axis[n], self.x_axis[j])
        
    
    def upwind(self):
        v = np.zeros([self.time_steps, self.space_steps])
        v[0, :] = self.exact_sol[0, :]
        for n in range(self.time_steps - 1):
            for j in range(1, self.space_steps):
                v[n+1, j] = v[n, j] - self.CFL * (v[n, j] - v[n, j-1])
            
        return v
    

    def centered(self):
        v = np.zeros([self.time_steps, self.space_steps])
        v[0, :] = self.exact_sol[0, :]
        for n in range(self.time_steps - 1):
            for j in range(1, self.space_steps-1):
                v[n+1, j] = v[n, j] - self.CFL/2 * (v[n, j+1] - v[n, j-1])
            
        return v
    

    def lax_wendroff(self):
        v = np.zeros([self.time_steps, self.space_steps])
        v[0, :] = self.exact_sol[0, :]
        for n in range(self.time_steps - 1):
            for j in range(1, self.space_steps-1):
                smoothing = (self.CFL**2 / 2 ) * (v[n, j+1] - 2 * v[n, j] + v[n, j-1])
                v[n+1, j] = v[n, j] - self.CFL/2 * (v[n, j+1] - v[n, j-1]) + smoothing
                
        return v

    
    def lax_friedrichs(self):
        v = np.zeros([self.time_steps, self.space_steps])
        v[0, :] = self.exact_sol[0, :]
        for n in range(self.time_steps - 1):
            for j in range(1, self.space_steps-1):
                v[n+1, j] = (v[n, j+1] + v[n, j - 1])/2 - (self.CFL/2)*(v[n, j+1]- v[n, j-1])
                
        return v

    def backward_euler(self, t0):
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
                op_mat[n, 0] = 1
                op_mat[n, 1] = -self.CFL/2
            elif n == self.time_steps - 1:
                op_mat[n, n - 1] = self.CFL/2
                op_mat[n, n] = 1
            else:
                op_mat[n, n - 1] = self.CFL/2
                op_mat[n, n] = 1
                op_mat[n, n + 1] = -self.CFL/2

        op_mat = np.linalg.inv(op_mat)
        for n in range(self.time_steps - 1):
            v[n+1, :] = np.matmul(op_mat, v[n, :])
        
        return v

    def crank_nicholson(self, t0):
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
                l_op_mat[n, 0] = 1
                l_op_mat[n, 1] = -self.CFL/4
            elif n == self.time_steps - 1:
                l_op_mat[n, n - 1] = self.CFL/4
                l_op_mat[n, n] = 1
            else:
                l_op_mat[n, n - 1] = self.CFL/4
                l_op_mat[n, n] = 1
                l_op_mat[n, n + 1] = -self.CFL/4

        #Setting up right side operations matrix
        for n in range(self.time_steps):
            if n == 0:
                r_op_mat[n, 0] = 1
                r_op_mat[n, 1] = self.CFL/4
            elif n == self.time_steps - 1:
                r_op_mat[n, n - 1] = -self.CFL/4
                r_op_mat[n, n] = 1
            else:
                r_op_mat[n, n - 1] = -self.CFL/4
                r_op_mat[n, n] = 1
                r_op_mat[n, n + 1] = self.CFL/4

        l_op_mat = np.linalg.inv(l_op_mat)
        for n in range(self.time_steps - 1):
            v[n+1, :] = np.matmul(np.matmul(l_op_mat, r_op_mat), v[n, :])

        return v
    
    def method_of_lines(self, q_num):
        A = np.zeros([self.space_steps, self.space_steps])
        w = np.zeros([self.time_steps, self.space_steps])
        w[0, :] = self.initial
        l = self.space_steps - 1
        if q_num == 6:
            for n in range(self.space_steps):
                if n == 0:
                    A[n, 1], A[n, 2], A[n, 3] = 45, -9, 1
                    A[n, l], A[n, l-1], A[n, l-2] = -45, 9, -1
                elif n == 1:
                    A[n, 2], A[n, 3], A[n, 4] = 45, -9, 1
                    A[n, 0], A[n, l], A[n, l-1] = -45, 9, -1
                elif n == 2:
                    A[n, 3], A[n, 4], A[n, 5] = 45, -9, 1
                    A[n, 1], A[n, 0], A[n, l] = -45, 9, -1
                elif n == l:
                    A[n, 0], A[n, 1], A[n, 2] = 45, -9, 1
                    A[n, l-1], A[n, l-2], A[n, l-3] = -45, 9, -1
                elif n == l-1:
                    A[n, l], A[n, 0], A[n, 1] = 45, -9, 1
                    A[n, l-2], A[n, l-3], A[n, l-4] = -45, 9, -1
                elif n == l-2:
                    A[n, l-1], A[n, l], A[n, 0] = 45, -9, 1
                    A[n, l-3], A[n, l-4], A[n, l-5] = -45, 9, -1
                else:
                    A[n, n+1], A[n, n+2], A[n, n+3] = 45, -9, 1
                    A[n, n-1], A[n, n-2], A[n, n-3] = -45, 9, -1
        elif q_num == 4:
            for n in range(self.space_steps):
                if n == 0:
                    A[n, 1], A[n, 2] = 8, -1
                    A[n, l], A[n, l-1] = -8, 1
                elif n == 1:
                    A[n, 2], A[n, 3] = 8, -1
                    A[n, 0], A[n, l] = -8, 1
                elif n == l:
                    A[n, 0], A[n, 1] = 8, -1
                    A[n, l-1], A[n, l-2] = -8, 1
                elif n == l-1:
                    A[n, l], A[n, 0] = 8, -1
                    A[n, l-2], A[n, l-3] = -8, 1
                else:
                    A[n, n+1], A[n, n+2] = 8, -1
                    A[n, n-1], A[n, n-2] = -8, 1
                    
        #4 Stage Runge-Kutta
        for n in range(self.time_steps-1):
            m1 = self.k * np.matmul(A, w[n, :])

            #Save computation for reuse
            a_pow2 = np.linalg.matrix_power(A, 2)
            mat_op2 = self.k * A + (self.k**2)/2 * a_pow2
            m2 = np.matmul(mat_op2, w[n, :])

            a_pow3 = np.matmul(a_pow2, A)
            mat_op3 = mat_op2 + (self.k**3)/4 * a_pow3
            m3 = np.matmul(mat_op3, w[n, :])

            a_pow4 = np.matmul(a_pow3, A)
            m4 = np.matmul(mat_op3*2 - self.k*A + (self.k**4)/4*a_pow4, w[n, :])

            w[n+1, :] = w[n, :] + 1/6 * (m1 + 2*m2 + 2*m3 + m4)
        
        return w

    def get_err(self, o, approx, t):
        diff = self.exact_sol[t, :] - approx[t, :]
        return np.linalg.norm(diff, o)






