import heat_eq
import transport_eq
import numpy as np
import matplotlib.pyplot as plt

def graph_sol(scheme, t1, t2, approx_sol, exact, x_axis_a, x_axis_e):
        '''
        Displays graph of approximated soluton vs exact solution at timestep t
        :param scheme: Name of scheme used, just used for the title of the plot
        :param t: The snapshot of the solution to be plotted
        :param approx_sol: Grid of solution approxmations
        :return: A matplotlib plot should appear
        '''

        y_axis_a = approx_sol[t1]
        plt.plot(x_axis_a, y_axis_a, label='Aprroximated Solution')
        plt.plot(x_axis_e, exact[t2], label='Exact Solution')
        plt.ylabel("u(x)")
        plt.xlabel("x")
        plt.title(scheme)
        plt.legend()
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

def main():
    def f_heat(x):
        return x - np.pi
        
    def f_transport(t, x):
        if (0 <=  x + t <= np.pi):
            return x + t
        else:
            return 2*np.pi - x - t

    k_arr = [1/30000]
    h_arr = [1/100]
    approxes = call_multiple_iterable('heat', h_arr, k_arr, "forward_euler", 2, f_heat)
    init_model = heat_eq.Solve_Heat_Eq(1/100, 1/30000, 2, f_heat)
    exact = init_model.Crank_Nicolson()
    for i, a in enumerate(approxes):
        space_steps = round((2*np.pi - 0) / h_arr[i])
        x_axis_a = np.linspace(0, 2*np.pi, space_steps)
        graph_sol("Forward Euler", 600-1, 600-1, a, exact, x_axis_a, init_model.x_axis)

if __name__ == '__main__':
    main()


