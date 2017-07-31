from dask.distributed import Client, Variable
import dask.bag as db
from operator import add
import argparse
import numpy as np
import scipy.linalg as sla
import functools
import os

###################################
# Utility functions
###################################
def select_topr(vct_input, r):
    """
    Returns the R-th greatest elements indices
    in input vector and store them in idxs_n.
    """
    temp = np.argpartition(-vct_input, r)
    idxs_n = temp[:r]
    return idxs_n

def input_to_rowmatrix(raw_rdd, indices, norm):
    """
    Utility function for reading the matrix data
    """
    p_and_n = functools.partial(parse_and_normalize, norm = norm)
    numpy_rdd = db.zip(indices, raw_rdd).map(lambda x: (x[0], p_and_n(x[1])))
    return numpy_rdd


###################################
# Helper functions
###################################

def parse_and_normalize(line, norm):
    """
    Utility function. Parses a line of text into a floating point array, then
    whitens the array.
    """
    x = np.array([float(c) for c in line.strip().split()])
    if norm:
        x -= x.mean()  # 0-mean.
        x /= sla.norm(x)  # Unit norm.
    return x

def vector_matrix(row):
    """
    Applies u * S by row-wise multiplication, followed by a reduction on
    each column into a single vector.
    """
    row_index, vector = row
    
    # Extract the broadcast variables.
    u =  _U_.get()

    # This means we're in the first iteration and we just want a random
    # vector. To ensure all the workers generate the same random vector,
    # we have to seed the RNG identically.
    if len(u) == 2:
        T, seed = u[0],u[1]
        np.random.seed(seed)
        u = np.random.random(T)
        u -= u.mean()
        u /= sla.norm(u)
    u = u[row_index]

    out = []
    for i in range(vector.shape[0]):
        out.append((i,u * vector[i]))
    return out

def matrix_vector(row):
    """
    Applies S * v by row-wise multiplication. No reduction needed, as all the
    summations are performed within this very function.
    """
    k, vector = row  

    # Extract the broadcast variables.
    v = _VI_.get()
    indices = _I_.get()

    # Perform the multiplication using the specified indices in both arrays.
    innerprod = np.dot(vector[indices], v)

    return [k, innerprod]

def deflate(row):
    """
    Deflates the data matrix by subtracting off the outer product of the
    broadcasted vectors and returning the modified row.
    """
    k, vector = row
    
    # Extract the broadcast variables.
    u = _UU_.get()
    indices = _I_.get()
    values = _VI_.get()
    values = np.array(values)
    
    vector[indices] -= (u[k] * values)
    return (k, vector)

def partition_reduction(total,x):
    return total + x[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'PySpark Dictionary Learning',
        add_help = 'How to use', prog = 'python R1DL_Spark.py <args>')

    # Inputs.
    parser.add_argument("-i", "--input", required = True,
        help = "Input file containing the matrix S.")
    parser.add_argument("-T", "--rows", type = int, required = True,
        help = "Number of rows (observations) in the input matrix S.")
    parser.add_argument("-P", "--cols", type = int, required = True,
        help = "Number of columns (features) in the input matrix S.")

    # Optional.
    parser.add_argument("-r", "--pnonzero", type = float, default = 0.07,
        help = "Percentage of non-zero elements. [DEFAULT: 0.07]")
    parser.add_argument("-m", "--dictatoms", type = int, default = 5,
        help = "Number of the dictionary atoms. [DEFAULT: 5]")
    parser.add_argument("-e", "--epsilon", type = float, default = 0.01,
        help = "The convergence criteria in the ALS step. [DEFAULT: 0.01]")
    parser.add_argument("--normalize", action = "store_true",
        help = "If set, normalizes input data.")
    parser.add_argument("--debug", action = "store_true",
        help = "If set, turns out debug output.")

    # Dask options.
    parser.add_argument("--partitions", type = int, default = None,
        help = "Number of RDD partitions to use. [DEFAULT: 4 * CPUs]")
    parser.add_argument("--execmem", default = "8g",
        help = "Amount of memory for each executor. [DEFAULT: 8g]")

    # Outputs.
    parser.add_argument("-d", "--dictionary", required = True,
        help = "Output path to dictionary file.(file_D)")
    parser.add_argument("-o", "--output", required = True,
        help = "Output path to z matrix.(file_z)")
    parser.add_argument("--prefix", required = True,
        help = "Prefix strings to the output files")

    args = vars(parser.parse_args())

    # Parse out the command-line arguments.
    file_s = args['input']
    T = args['rows']
    P = args['cols']

    epsilon = args['epsilon']            # convergence stopping criterion
    M = args['dictatoms']                # dimensionality of the learned dictionary
    R = int(args['pnonzero'] * P)        # enforces sparsity
    u_new = np.zeros(T)                  # atom updates at each iteration
    v = np.zeros(P)
    max_iterations = P * 10

    # In cluster mode, pass the address:port of the Scheduler 
    client = Client()
    
    # Read a text file into a Dask bag ~ Spark RDD
    input = db.read_text(file_s)

    #Apparently in Bags, you need another Bag which has the indices to zip with. Hence, creating indices for our data
    l = db.from_sequence(range(input.count()), npartitions = 1)

    S = input_to_rowmatrix(input, l, True)

    #Global/Broadcast Variables
    _U_ = Variable('_U_')
    _UU_ = Variable('_UU_')
    _I_ = Variable('_I_')
    _VI_ = Variable('_VI_')
    
    file_D = os.path.join(args['dictionary'], "{}_D.txt".format(args["prefix"]))
    file_z = os.path.join(args['output'], "{}_z.txt".format(args["prefix"]))

    #Start the loop!
    for m in range(M):
        print ('M: '+str(m))
        seed = np.random.randint(max_iterations + 1, high = 4294967295)
        np.random.seed(seed)
        u_old = np.random.random(T)
        num_iterations = 0
        delta = 2 * epsilon

        while num_iterations < max_iterations and delta > epsilon:
            _U_.set(list(u_old)) if num_iterations > 0 else _U_.set(list([T, seed]))

            v = S.map(vector_matrix).flatten().foldby(lambda (key, value): key, partition_reduction, 0, combine=add).compute()
            r = client.gather(v)
            v = np.take(sorted(r), indices = 1, axis = 1)
            
            indices = np.sort(select_topr(v, R))
            _I_.set(list(indices))
            _VI_.set(list(v[indices]))

            u_newt = S.map(matrix_vector).compute()
            u_new = client.gather(u_newt)
            u_new = np.take(sorted(u_new), indices = 1, axis = 1)
           
            # Subtract off the mean and normalize.
            u_new -= u_new.mean()
            u_new /= sla.norm(u_new)

            # Update for the next iteration.
            delta = sla.norm(u_old - u_new)
            u_old = u_new
            num_iterations = num_iterations+1
        


        with open(file_D, "a+") as fD:
            np.savetxt(fD, u_new, fmt = "%.6f", newline = " ")
            fD.write("\n")

        temp_v = np.zeros(v.shape)
        temp_v[indices] = v[indices]
        v = temp_v
        with open(file_z, "a+") as fz:
            np.savetxt(fz, v, fmt = "%.6f", newline=" ")
            fz.write("\n")    
        
        _UU_.set(list(u_new))
        _VI_.set(list(v[indices]))
        _I_.set(list(indices))
        S = S.map(deflate)
        
 
    