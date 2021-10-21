import numpy as np

class Solve_Viscid_Burgers:
    def __init__(self, h, k, tf, f, eta, t0=0, x0=0, xf=2*np.pi):
        self.h = h
        self.k = k
        self.f = f
        self.eta = eta
        self.time_steps = round((tf - t0) / k)
        self.space_steps = round((xf - x0) / h)
        self.x_axis = np.linspace(x0, xf, self.space_steps)
        self.t_axis = np.linspace(t0, tf, self.time_steps)
        self.sig = k/(h**2)
        print(self.sig)
        self.initial = [f(x) for x in self.x_axis]
        self.exact_sol = np.zeros([self.time_steps, self.space_steps])

    def set_exact(self, exact):
        self.exact_sol = exact
    
    def Fully_Implicit(self):
        #Initializing grid
        v = np.zeros([self.time_steps, self.space_steps])
        v[0, :] = self.initial

        #Setting up left side operations matrix
        cof = self.eta * self.sig
        half_lam = self.k / (2 * self.h)

        for t in range(self.time_steps - 1):
            op_mat = np.zeros([self.space_steps, self.space_steps])
            for n in range(self.space_steps):
                if n == 0:
                    op_mat[n, self.space_steps - 1] = -cof - half_lam * v[t, 0]
                    op_mat[n, 0] = 1 + 2*cof
                    op_mat[n, 1] = -cof + half_lam * v[t, 0]
                elif n == self.space_steps - 1:
                    op_mat[n, n - 1] = -cof - half_lam * v[t, n]
                    op_mat[n, n] = 1 + 2*cof
                    op_mat[n, 0] =  -cof + half_lam * v[t, n]
                else:
                    op_mat[n, n - 1] = -cof - half_lam * v[t, n]
                    op_mat[n, n] = 1 + 2*cof
                    op_mat[n, n + 1] = -cof + half_lam * v[t, n]
            op_mat = np.linalg.inv(op_mat)

            v[t+1, :] = np.matmul(op_mat, v[t, :])
        
        return v

    def Semi_Implicit(self):
        cof = self.eta * self.sig
        half_lam = self.k / (2 * self.h)

        #Initializing grid and operations matrix
        op_mat = np.zeros([self.space_steps, self.space_steps])
        v = np.zeros([self.time_steps, self.space_steps])
        v[0, :] = self.initial

        #Fill out and invert operations matrix
        for n in range(self.space_steps):
            if n == 0:
                op_mat[n, 1] = op_mat[n, self.space_steps - 1] = -cof
            elif n == self.space_steps - 1:
                op_mat[n, n - 1] = op_mat[n, 0] =  -cof
            else:
                op_mat[n, n - 1] = op_mat[n, n + 1] = -cof
            op_mat[n, n] = 1 + 2*cof
        op_mat = np.linalg.inv(op_mat)

        for t in range(self.time_steps - 1):
            right = np.ones(self.space_steps)
            for j in range(self.space_steps):
                if j == 0:
                    r = v[t, j] + half_lam * v[t, j] * (v[t, j + 1] - v[t, self.space_steps-1])
                elif j + 1 == self.space_steps:
                    r = v[t, j] + half_lam * v[t, j] * (v[t, 0] - v[t, j-1])
                else:
                    r = v[t, j] + half_lam * v[t, j] * (v[t, j+1] - v[t, j-1])
                right[j] = r
            v[t+1, :] = np.matmul(op_mat, right)

        return v
