import numpy as np # solving linear equations
from Equation import Expression # parsing functions from strings
import matplotlib.pyplot as plt # plotting
import matplotlib.ticker as plticker
from math import floor, ceil

# Defaults
np.set_printoptions(suppress=True)
plt.style.use('seaborn-white')
__round_val__ = 3

def get_inner_points(boundary_points, h, k):
    
    boundary_points_xy = list(boundary_points.keys())    
    unique_y = list(set([coordinates[1] for coordinates in boundary_points_xy ]))    
    inner_points = list()

    for y in unique_y:
        x_vals = [pt[0] for pt in boundary_points_xy if pt[1] == y]    
        smallest_x = min(x_vals)
        biggest_x = max(x_vals)

        if ( smallest_x % h != 0 ):
            if (smallest_x < 0):
                xi = ceil(smallest_x/h)+h
            else:
                xi = floor(smallest_x/h)+h
        else:
            xi = smallest_x + h

        while xi < biggest_x:
            point = (xi, y)

            if ( point not in boundary_points_xy ):
                inner_points.append(point)

            xi = round(xi+h,__round_val__)
    
    return inner_points

def get_system_equations(inner_points, boundary_points, h, k, uxx_coeff_fn, uyy_coeff_fn, eqn_fn):

    unknowns = len(inner_points)
    mat = np.zeros([unknowns, unknowns])
    vals = np.zeros([unknowns, 1])

    # Filling the matrix with the equations
    for i, point in enumerate(inner_points):

        alpha1 = 1
        alpha2 = 1
        beta1 = 1
        beta2 = 1

        top_point = (point[0], round(point[1] + k, __round_val__))
        bottom_point = (point[0], round(point[1] - k,__round_val__))
        left_point = (round(point[0] - h,__round_val__), point[1])
        right_point = (round(point[0] + h,__round_val__), point[1])

        val_x = 2*uxx_coeff_fn(point[0], point[1])/(h**2)
        val_y = 2*uyy_coeff_fn(point[0], point[1])/(k**2)
        vals[i] = eqn_fn(point[0], point[1])
        
        # Get boundary points
        if (top_point not in inner_points):
            is_irregular_top_point = True
            for pt in boundary_points:
                if (pt[0] == point[0] and pt[1] > point[1]):
                    top_point = pt
                    break
        else:
            is_irregular_top_point = False

        if (bottom_point not in inner_points):
            is_irregular_bottom_point = True
            for pt in boundary_points:
                if (pt[0] == point[0] and pt[1] < point[1]):
                    bottom_point = pt  
                    break
        else:
            is_irregular_bottom_point = False

        if (right_point not in inner_points):
            is_irregular_right_point = True
            for pt in boundary_points:
                if (pt[1] == point[1] and pt[0] > point[0]):
                    right_point = pt
        else:
            is_irregular_right_point = False

        if (left_point not in inner_points):
            is_irregular_left_point = True
            for pt in boundary_points:
                if (pt[1] == point[1] and pt[0] < point[0]):
                    left_point = pt   
        else:
            is_irregular_left_point = False
        

        # Fill matrices
        if ( is_irregular_top_point and is_irregular_bottom_point): 
            beta1 = abs(top_point[1] - point[1])/k
            beta2 = abs(point[1] - bottom_point[1])/k

            vals[i] += -val_y*( (boundary_points[top_point]/(beta1*(beta1+beta2))) + 
                                                   (boundary_points[bottom_point]/(beta2*(beta1+beta2))) )

        elif ( is_irregular_top_point ):
            beta1 = abs(top_point[1] - point[1])/k
            vals[i] += -val_y*(boundary_points[top_point]/(beta1*(1+beta1)))
            mat[i][inner_points.index(bottom_point)] = val_y/(1+beta1)

        elif ( is_irregular_bottom_point ):
            beta2 = abs(point[1] - bottom_point[1])/k
            vals[i] += -val_y*(boundary_points[bottom_point]/(beta2*(1+beta2)))
            mat[i][inner_points.index(top_point)] = val_y/(1+beta2)

        else:
            mat[i][inner_points.index(bottom_point)] = val_y/2
            mat[i][inner_points.index(top_point)] = val_y/2


        if ( is_irregular_right_point and is_irregular_left_point):
            alpha1 = abs(point[0] - left_point[0])/h
            alpha2 = abs(right_point[0] - point[0])/h

            vals[i] += -val_x*( (boundary_points[left_point]/(alpha1*(alpha1+alpha2))) +
                                                            (boundary_points[right_point]/(alpha2*(alpha1+alpha2))) )
        elif ( is_irregular_left_point ):
            alpha1 = abs(point[0] - left_point[0])/h
            vals[i]  +=  -val_x*(boundary_points[left_point]/(alpha1*(alpha1+1)))
            mat[i][inner_points.index(right_point)] = val_x*(1/(1+alpha1))

        elif ( is_irregular_right_point ):
            alpha2 = abs(right_point[0] - point[0])/h
            vals[i]  +=  -val_x*(boundary_points[right_point]/(alpha2*(alpha2+1)))
            mat[i][inner_points.index(left_point)] = val_x*(1/(1+alpha2))

        else:
            mat[i][inner_points.index(left_point)] = val_x/2
            mat[i][inner_points.index(right_point)] = val_x/2

        mat[i][i] = -(1/(beta1*beta2))*val_y + (-1/(alpha1*alpha2))*val_x

    return vals, mat


def PDE_Irregular_Boundaries(boundary_points, h, k, uyy_coeff_fn, uxx_coeff_fn, eqn_fn):

    """
    Solves a second-order PDE with regular/irregular boundaries.
    
    Important Notes:
    You must pass all boundary points values that intersects with the grid 
    the equation passed will be in the form: uyy_coeff_fn*uyy + uxx_coeff_fn*uxx = eqn_Fn
    
    
    Parameters: 
    boundary_points (dict): { (x, y) : initial_value} 
    h (float):              delta x-axis
    k (float):              delta y-axis
    uyy_coeff_fn (str):     string represenetation of the coeeficient function of the second-order partial derivative w.r.t y
    uxx_coeff_fn (str):     string represenetation of the coeeficient function of the second-order partial derivative w.r.t x
  
    Returns: 
    values: dict with the values of points inside the boundary
    
    """

    # Parsing equations to functions
    try:
        uyy_coeff_fn = Expression(uyy_coeff_fn,["x","y"])
        uxx_coeff_fn = Expression(uxx_coeff_fn,["x","y"])
        eqn_fn = Expression(eqn_fn,["x","y"])
        
    except Exception:
        print("Error: Please enter a valid equation format in terms of x and y")
        return
    
    inner_points = get_inner_points(boundary_points, h, k)
    vals, mat = get_system_equations(inner_points, boundary_points, h, k, uxx_coeff_fn, uyy_coeff_fn, eqn_fn)

    # Solving equations
    invMat = np.linalg.inv(mat)
    x = np.dot(invMat,vals)

    # Formating the output dictionary
    values = dict()
    for i in range(len(x)):
        values[inner_points[i]] = round(x[i][0], __round_val__)

    return values

 
def Plot_results(inner_points, boundary_points, h, k):
    
    plot_pt = {**inner_points, **boundary_points}
    fig = plt.figure(figsize=(12, 12))
    ax = fig.subplots()

    xy = list(plot_pt.keys())
    values = list(plot_pt.values())
    ax.scatter(*zip(*xy))

    # Grid
    plt.grid()
    xloc = plticker.MultipleLocator(base=h)
    yloc = plticker.MultipleLocator(base=k)
    ax.xaxis.set_major_locator(xloc)
    ax.yaxis.set_major_locator(yloc)
    ax.yaxis.grid(linestyle='-.')
    ax.xaxis.grid(linestyle='-.')

    # Annotations
    for i, xy in enumerate(xy):                                       
        ax.annotate('{:.4f}'.format(values[i]), xy=xy, textcoords='data', fontsize=13) 

    plt.show()
    
    
    
def PDE_irregular_boundaries_interface():
    try:
        #Inputs
        points = dict()

        print("Welcome to our PDE calculator 😁")
        print("---------------------------------\n")
        
        print("You'll now enter all points of the boundaries that intersects with the grid, one by one and the initial value at this point")
        num = int(input("Enter the number of points: "))
        
        for i in range(num):
            x = float(input("Enter the x value of the {} point: ".format(i+1)))
            y = float(input("Enter the y value of the {} point: ".format(i+1)))
            value = float(input("Enter the initial value at the {} point: ".format(i+1)))
            points[(x,y)] = value
            
        h = float(input("Enter the value of h: "))
        k = float(input("Enter the value of k: "))
        uyy = str(input("Enter the value of u<yy> coefficient as a function in terms of x and y e.g. y*sin(x): ")) 
        uxx = str(input("Enter the value of u<xx> coefficient as a function in terms of x and y e.g. y*sin(x): "))
        eqn_fn = str(input("Enter the value of the equation's function in terms of x and y: "))
        
        print("---------------------------------\n")
        print("The magic is done ✅")
        
        # Calculation and plotting
        Results = PDE_Irregular_Boundaries(boundary_points = points, h = h, k = k, uyy_coeff_fn = uyy, 
                                         uxx_coeff_fn = uxx, eqn_fn = eqn_fn)

        Plot_results(Results, boundary_points = points, h= h, k=h )

    except:
        print("Oops! 🤕, There's an error please double check the inputs again")