from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import sys, yaml


if __name__ == '__main__':
    with open('rendered_wano.yml') as file:
        wano_file = yaml.full_load(file)
    
    no_plots = len(wano_file["Parameters"])
    print(wano_file["Parameters"][1]['Beta'])

    n_mesh = []
    n_funcspace = []
    beta = []
    R0 = []
    label_var = []

    for ii in range(no_plots):
        n_mesh.append(int(wano_file["Parameters"][ii]['Mesh']))
        n_funcspace.append(int(wano_file["Parameters"][ii]['FunctionSpace']))
        beta.append(float(wano_file["Parameters"][ii]['Beta']))
        R0.append(float(wano_file["Parameters"][ii]['R0']))
        label_var.append(str(wano_file["Parameters"][ii]['label']))

    print(n_mesh)
    for ii in range(no_plots):
        domain = Circle(Point(0, 0), 1)
        mesh = generate_mesh(domain, n_mesh[ii])
        V = FunctionSpace(mesh, 'P', n_funcspace[ii])
        # Define boundary condition
        w_D = Constant(0)

        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, w_D, boundary)

        #Define load
        #beta = 1
        #R0 = 0.6
        p = Expression('4*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R0, 2)))',
                degree=1, beta=beta[ii], R0=R0[ii])
        p_e = p

        # Define variational problem
        w = TrialFunction(V)
        v = TestFunction(V)
        a = dot(grad(w), grad(v))*dx
        L = p*v*dx

        # Compute solution
        w = Function(V)
        solve(a == L, w, bc)

        # Plot solution
        p = interpolate(p, V)
        #plot(w, title='Deflection')
        #plot(p, title='Load')

        #  Save solution to file in VTK format
        vtkfile_w = File('poisson_membrane/deflection.pvd')
        vtkfile_w << w
        vtkfile_p = File('poisson_membrane/load.pvd')
        vtkfile_p << p

        # Curve plot along x = 0 comparing p and w
        tol = 0.001  # avoid hitting points outside the domain
        y = np.linspace(-1 + tol, 1 - tol, 101)
        points = [(0, y_) for y_ in y]  # 2D points
        w_line = np.array([w(point) for point in points])
        p_line = np.array([p(point) for point in points])
        plt.plot(y, p_line, '-o', linewidth=2)
        plt.plot(y, 50*w_line, '--', linewidth=2)  # magnify w
        

    doubled_legend = [item for sub in label_var for item in [sub] * 2]
    plt.legend(doubled_legend)
    plt.grid(True)
    plt.title('Membrane Deflection')
    plt.xlabel('$y$')
    plt.ylabel("Deflection ($\\times 50$) (--) " + " and " " Load (o)")
    #plt.legend(['Deflection ($\\times 50$)', 'Load'], loc='upper left')
    plt.savefig('curves.png', dpi = 200)

    # Compute error in L2 norm
    error_L2 = errornorm(p_e, p, 'L2')
    f = open( 'OUTFE', 'w' )
    f.write( "mesh  " + str(n_mesh)+'\n')
    f.write( "funcspace  " + str(n_funcspace)+'\n')
    f.write( "beta  " + str(beta)+'\n')
    f.write( "R0 " + str(R0)+'\n')
    f.write( "error  " + str(error_L2))

    f.close()