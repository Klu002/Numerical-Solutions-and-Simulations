import numpy as np
import matplotlib.pyplot as plt

class Solve_Wave_Eq:

    def __init__(self, h, k, tf, f, t0=0, x0=0, xf=2*np.pi):
        self.h = h
        self.k = k
        self.f = f
        self.time_steps = round((tf - t0) / k)
        self.space_steps = round((xf - x0) / h)
        self.x_axis = np.linspace(x0, xf, self.space_steps)
        self.t_axis = np.linspace(t0, tf, self.time_steps)
        self.CFL = k/h
        self.exact_sol = np.zeros([self.time_steps, self.space_steps])


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
    
    def get_err(self, o, approx, t):
        diff = self.exact_sol[t, :] - approx[t, :]
        return np.linalg.norm(diff, o)

    def graph_sol(self, scheme: str, t: int, approx_sol):
        '''
        Displays graph of approximated soluton vs exact solution at timestep t
        :param scheme: Name of scheme used, just used for the title of the plot
        :param t: The snapshot of the solution to be plotted
        :param approx_sol: Grid of solution approxmations
        :return: A matplotlib plot should appear
        '''
        y_axis = approx_sol[t]
        plt.plot(self.x_axis, y_axis, label='Aprroximated Solution')
        plt.plot(self.x_axis, self.exact_sol[t], label='Exact Solution')
        plt.ylabel("u(x)")
        plt.xlabel("x")
        plt.title(scheme)
        plt.legend()
        plt.show()

def call_multiple_h(model, h_arr, k_arr, t, scheme):
    '''
    Calls a scheme for each h, returns a plot at the specified time step
    :model: Solve_Wave_Eq object
    :params h_arr: Array of h values
    :params k_arr: Array of k values
    :params t: Time step to print at
    :params scheme: Scheme to keep on calling
    '''

    for i, h in enumerate(h_arr):
        model.change_h(h)
        model.change_k(k_arr[i])
        if scheme == 'upwind':
            approx = model.upwind()
        elif scheme == 'centered':
            approx = model.centered()
        elif scheme == 'lax-wendroff':
            approx = model.lax_wendroff()
        elif scheme == 'lax-friedrichs':
            approx = model.lax_friedrichs()
        elif scheme == 'backward euler':
            approx = model.backward_euler()
        elif scheme == 'crank-nicholson':
            approx = model.crank_nicholson()
        else:
            raise Exception('Invalid scheme')
        model.graph_sol(scheme, t, approx)
        print(scheme + " L2 Error: "  + str(model.get_err(2, approx, 2**8 - 1)))
        print(scheme + " Max Error: "  + str(model.get_err(float('inf'), approx, 2**8 - 1)))

def main():
    def f(t, x):
        if (0 <=  x + t <= np.pi):
            return x + t
        else:
            return 2*np.pi - x - t
    
    

    
    '''
    maxes = []
    for n in range(approx.shape[0]):
        maxes.append(np.linalg.norm(approx[n, :], float('inf')))
    plt.plot(range(approx.shape[0]), maxes)
    plt.ylabel(r'Max($v(x)_n$)')
    plt.xlabel('Time step')
    plt.title(r'Naive Centered Scheme, h = $(\frac{1}{2})^7$')
    plt.show()
    '''
    

if __name__ == '__main__':
    main()






