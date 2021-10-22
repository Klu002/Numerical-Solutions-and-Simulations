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

    #init_model = heat_eq.Solve_Heat_Eq(1/100, 1/300, 2, f_heat)
    #exact= init_model.Crank_Nicolson()
    #model = heat_eq.Solve_Heat_Eq(1/100, 1/30000, 2, f_heat)
    #df = model.Dufort_Frankel()
    #be = model.Backward_Euler()
    #fe = model.Forward_Euler()
    #cn = model.Crank_Nicolson()

    gdir = 'Example_Solution_Graphs/'
    #graph_sol(r'Solutions for $\frac{k}{h^{2}} = \frac{1}{3}$', "Dufort_Frankel", 60000-1, df, model.x_axis, x_axis_e=[model.x_axis, model.x_axis, model.x_axis], t2=[60000-1, 60000-1, 60000-1], other_sols=[be, cn, fe], other_sol_name=["Backward Euler", "Crank Nicholson", "Forward Euler"], save_dir=gdir + "HEhk2.png")
    #graph_sol(r'$\frac{k}{h^{2}} = \frac{1}{3}$', "Backward Euler", [60000-1], be, model.x_axis, save_dir=gdir + "BE30000.png")
    #graph_sol(r'$\frac{k}{h^{2}} = \frac{1}{3}$',"Crank Nicolson", [60000-1], cn, model.x_axis, save_dir=gdir + "CN30000.png")
    #graph_sol(r'$\frac{k}{h^{2}} = \frac{1}{3}$',"Dufort Frankel", [60000-1], df, model.x_axis, save_dir=gdir + "DF30000.png")
    #graph_sol(r'$\frac{k}{h^{2}} = \frac{1}{3}$',"Forward Eueler", [60000-1], fe, model.x_axis, save_dir=gdir + "FE30000.png")
    #graph_sol("Forward Euler", [600-1], be, model.x_axis, x_axis_e=model.x_axis, t2=600-1, exact=exact)#, save_dir=gdir + "FEK30KPickled.png")

    #print("Dufort norm: {}".format(np.linalg.norm(df[6000-1]-exact[6000-1], 2)))
    #print("Backward Euler norm: {}".format(np.linalg.norm(be[6000-1]-exact[6000-1], 2)))

    '''
    model1 = heat_eq.Solve_Heat_Eq(1/100, 1/3000, 2, f_heat)
    df1 = model1.Dufort_Frankel()
    be1 = model1.Backward_Euler()

    graph_sol("Dufort_Frankel", [6000-1], df1, model.x_axis, x_axis_e=model1.x_axis, t2=6000-1, exact=exact)#, save_dir=gdir + "FEK30KPickled.png")
    graph_sol("Backward_Euler", [6000-1], be1, model.x_axis, x_axis_e=model1.x_axis, t2=6000-1, exact=exact)#, save_dir=gdir + "FEK30KPickled.png")
    print("Dufort norm: {}".format(np.linalg.norm(df[600-1]-exact[600-1], 2)))
    print("Backward Euler norm: {}".format(np.linalg.norm(be[599]-exact[599], 2)))

    model = heat_eq.Solve_Heat_Eq(1/100, 1/30000, 2, f_heat)
    df = model.Dufort_Frankel()
    be = model.Backward_Euler()

    graph_sol("Dufort_Frankel", [600-1], df, model.x_axis, x_axis_e=model.x_axis, t2=60000-1, exact=exact)#, save_dir=gdir + "FEK30KPickled.png")
    graph_sol("Backward_Euler", [600-1], be, model.x_axis, x_axis_e=model.x_axis, t2=60000-1, exact=exact)
    graph_sol("Forward_Euler", [600-1], be, model.x_axis, x_axis_e=model.x_axis, t2=60000-1, exact=exact)#, save_dir=gdir + "FEK30KPickled.png")
    print("Dufort norm: {}".format(np.linalg.norm(df[600-1]-exact[600-1], 2)))
    print("Backward Euler norm: {}".format(np.linalg.norm(be[599]-exact[599], 2)))
    '''
    '''
    model1 = heat_eq.Solve_Heat_Eq(1/100, 1/30000, 2, f_heat)
    fe1 = model1.Forward_Euler()
    df1 = model1.Dufort_Frankel()
    be1 = model1.Backward_Euler()

    model2 = heat_eq.Solve_Heat_Eq(1/100, 1/3000, 2, f_heat)
    fe2 = model2.Forward_Euler()
    df2 = model2.Dufort_Frankel()
    be2 = model2.Backward_Euler()
    '''

    #pickle_sol('EXACT', 'Pickled_Solutions/', 'in', exact)
    
    #model = second_order_wave.Solve_Second_Order_Wave(1/100, 1/50, 1, np.sin, np.cos, 1, exact=f_wave)
    #approx1 = model.Newmark()
    #model.get_exact()
    #exact = model.exact_sol
    #approx = model.Newmark()
    #approx1 = pickle_sol('NMk=200.pickle', 'Pickled_Solutions/', 'out')
    #print("Newmark norm: {}".format(np.linalg.norm(approx1[49]-exact[49], 2)))
    #print("Leapfrog norm: {}".format(np.linalg.norm(approx2[49]-exact[49], 2)))
    #plt.plot(model.x_axis, approx1[49], label='Newmark Method')
    #plt.plot(model.x_axis, approx2[49], label='Leap Frog Method')
    #plt.plot(model.x_axis, exact[49], linestyle='dashed', label='Exact Solution')

    #plt.ylabel('u(x)')
    #plt.xlabel('x')
    #plt.legend()
    #plt.title(r"Second Order Wave Soluions T=1, $\frac{k}{h} = 2$")
     
    #diff1 = np.abs(approx1 - exact)
    #diff2 = np.abs(approx2 - exact)
    #plt.plot(model.x_axis, diff1[199])
    #plt.plot(model.x_axis, diff2[199])
    #plt.show()

    #model = heat_eq.Solve_Heat_Eq(1/100, 1/30000, 2, f_heat)
    #approx = model.Forward_Euler()
    #pickle_sol('FEk=30k', 'Pickled_Solutions/', 'in', approx)
    #exact = pickle_sol('BEk=30k', 'Pickled_Solutions/', 'out')
    #approx = pickle_sol('FEk=30k', 'Pickled_Solutions/', 'out')
    #
    #graph_sol("Forward_Euler", [60000-1], approx, model.x_axis, x_axis_e=model.x_axis, t2=60000-1)#, exact=exact), save_dir=gdir + "FEK30KPickled.png")
    
    model = viscid_burgers.Solve_Viscid_Burgers(1/100, 1/300, 50, f_heat, 0.005)
   # approx1= model.Fully_Implicit()
    approx2= model.Semi_Implicit()
    gdir = 'Example_Solution_Graphs/'
    plt.plot(model.t_axis, np.linalg.norm(approx2, axis=1))
    plt.show()
    #graph_sol(r"Viscid Burgers $\eta=0.005$", "Fully Implicit", 15000-1, approx1, model.x_axis, save_dir=gdir + 'VBFIT50ETAVERYSMALLFULLY.png')
    #graph_sol(r"Viscid Burgers $\eta=0.005$", "Semi Implicit", 15000-1, approx2, model.x_axis, save_dir=gdir + 'VBFIT50ETAVERYSMALLSEMI.png')
    #graph_sol(r"Viscid Burgers $\eta=1$", "Fully Implicit", 15000-1, approx1, model.x_axis, x_axis_e=[model.x_axis], t2=[15000-1], other_sols=[approx2], other_sol_name=["Semi Implicit"], save_dir=gdir + 'VBFIT50ETASMALL.png')
    '''
    model = transport_eq.Solve_Transport_Eq(1/100, 1/300, 1, f_heat, sol_f=f_exact_transport)
    model.exact()
    approx = model.method_of_lines(6)
    gdir = 'Example_Solution_Graphs/'
    graph_sol("Transport Equation Method of Lines Q6", "Approximated Solution", 299, approx, model.x_axis, x_axis_e=[model.x_axis], t2=[299], other_sols=[model.exact_sol], other_sol_name=["Exact Solution"], save_dir=gdir + 'TEMOLT1Q6.png')
    
    model = second_order_wave.Solve_Second_Order_Wave(1/100, 1/200, 1, np.sin, np.cos, 1)
    approx = model.Leap_Frog()
    plt.plot(model.x_axis, approx[199])
    plt.show()

    k_arr = [1/30000]
    h_arr = [1/100]
    approxes = call_multiple_iterable('heat', h_arr, k_arr, "forward_euler", 2, f_heat)
    init_model = heat_eq.Solve_Heat_Eq(1/100, 1/30000, 2, f_heat)
    exact = init_model.Crank_Nicolson()
    for i, a in enumerate(approxes):
        space_steps = round((2*np.pi - 0) / h_arr[i])
        x_axis_a = np.linspace(0, 2*np.pi, space_steps)
        graph_sol("Forward Euler", 600-1, 600-1, a, exact, x_axis_a, init_model.x_axis)
    '''
if __name__ == '__main__':
    main()


