from numpy.lib.type_check import nan_to_num
import heat_eq
import transport_eq
import viscid_burgers
import second_order_wave
import numpy as np
import matplotlib.pyplot as plt
import pickle

def pickle_sol(filename, pickle_dir, direction: str, solution=None):
    if direction == 'in':
        assert(solution is not None)
        outfile = open(pickle_dir + filename, 'wb')
        pickle.dump(solution, outfile)
        outfile.close()
        return None
    elif direction == 'out':
        assert(solution is None)
        infile = open(pickle_dir + filename, 'rb')
        ret = pickle.load(infile)
        infile.close()
        return ret
    else:
        raise ValueError('Invalid Direction')


def graph_sol(title, scheme, t, approx_sol, x_axis_a, x_axis_e=None, t2=None, other_sols=None, other_sol_name=None, save_dir=None):
        '''
        Displays graph of approximated soluton vs exact solution at timestep t
        :param scheme: Name of scheme used, just used for the title of the plot
        :param t: The snapshot of the solution to be plotted
        :param approx_sol: Grid of solution approxmations
        :return: A matplotlib plot should appear
        '''

        y_axis_a = approx_sol[t]
        plt.plot(x_axis_a, y_axis_a.T, label=scheme)
        if other_sols is not None:
            for i, other_sol in enumerate(other_sols):
                plt.plot(x_axis_e[i], other_sol[t2[i]], linestyle='dashed', label=other_sol_name[i])
        plt.ylabel("u(x)")
        plt.xlabel("x")
        plt.title(title)
        plt.legend()
        if save_dir:
            plt.savefig(save_dir, format='png')
        plt.show()
    
def call_multiple_iterable(eq, h_arr, k_arr, scheme, tf, f, t0=0, x0=0, xf=2*np.pi):
    approximations = []
    for i, h in enumerate(h_arr):
        k = k_arr[i]

        if eq == "heat":
            model = heat_eq.Solve_Heat_Eq(h, k, tf, f, t0, x0, xf)

            if scheme == 'forward_euler':
                approx = model.Forward_Euler()
            elif scheme == 'dufort_frankel':
                approx = model.Dufort_Frankel()
            else:
                raise Exception('Heat eq: Invalid scheme')

        elif eq == "transport":
            model = transport_eq.Solve_Transport_Eq(h, k, tf, f, t0, x0, xf)

            if scheme == 'upwind':
                approx = model.upwind()
            elif scheme == 'centered':
                approx = model.centered()
            elif scheme == 'lax-wendroff':
                approx = model.lax_wendroff()
            elif scheme == 'lax-friedrichs':
                approx = model.lax_friedrichs()
            raise Exception('Transport eq: Invalid scheme')

        else:
            raise Exception('Invalid Equation')
        approximations.append(approx)

    return approximations

def h_norm(approx, exact, h, h2=None):
    grid_points = []
    for n,v in enumerate(approx):
        if h2 is not None:
            #If we have a second h2, we convert to the equivilent index in the exact solution
            n2 = int(n*h / h2)
            if n == len(approx)-1 and n2 < len(exact) - 1:
                grid_points.append(v-exact[-1])
            else:
                grid_points.append(v-exact[n2])
        else:
            #If we do not, then exact is a function so we evaluate the difference at that space step
            grid_points.append(v-exact(n*h))
    return np.linalg.norm(grid_points)

def main():
    #Example initial conditions and exact solutions
    def f_heat(x):
        return x - np.pi

    def f_exact_transport(t, x):
        return (x+t)%(2*np.pi) - np.pi 

    def f_transport(t, x):
        if (0 <=  x + t <= np.pi):
            return x + t
        else:
            return 2*np.pi - x - t
    
    def f_wave(t, x):
        return np.sin(x + t)

    gdir = 'Example_Solution_Graphs/'

if __name__ == '__main__':
    main()


