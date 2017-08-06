import argparse
import numpy as np
from dask.distributed import Client, Variable
import dask.array as da
import operator

def op_selectTopR(vct_input, R):
    """
    Returns the R greatest elements

    parameters
    ----------
    vct_input : array, shape (T)
        indicating the input vector which is a
        vector we aimed to find the R
    R : integer
        indicates R greatest elemnts which we
        are seeking for.

    Returns
    -------
    max_values: array, shape (R)
        a vector with values for the R greatest elements
       
    """
    max_values = vct_input.topk(R)
    return max_values

def op_getResidual(S, u, v, idxs_n):
    """
    Returns the new S matrix by calculating :
        S =( S - uv )
    Here the product operation between u and v
    is an outer product operation.

    parameters
    ----------
    S : array, shape (T, P)
        The input matrix ( befor we stored the input
        file in this matrix at the main module of program)
        Here, we need to update this matrix for next iteration.
    u : array, shape (T)
        indicating 'u_new' vector (new vector
        of dictionary elements which will be used
        for updating the S matrix)
    v : array, shape (P)
        indicating 'v' vector ( which would be
        finally our output vector but here we are using
        this vector for updating S matrix by applying
        outer product of specific elements of v
        and u_new )
    idxs_n : array, shape (R)
        which is a vector encompassing Rth
        greatest elements indices.

    Returns
    -------
    S : array, shape (T, P)
        new S matrix based on above mentioned equation
        (updating S matrix for next iteration)
    """
    v_sparse = np.zeros(v.shape[0], dtype = np.float)
    v_sparse[idxs_n] = v[idxs_n]
    v_sparse = da.from_array(v_sparse, chunks=(1,))
    S = S - da.atop(operator.mul, 'ij', u, 'i', v_sparse, 'j', dtype='f8') 
    return S

def r1dl(S, nonzero, atoms, epsilon):
    """
    R1DL dictionary method.

    Parameters
    ----------
    S : array, shape (T, P)
        Input data: P instances, T features.
    nonzero : float
        Sparsity of the resulting dictionary (percentage of nonzero elements).
    atoms : integer
        Number of atoms in the resulting dictionary.
    epsilon : float
        Convergence epsilon in determining each dictionary atom.

    Returns
    -------
    D : array, shape (M, T)
        Dictionary atoms.
    Z : array, shape (M, P)
        Loading matrix.
    """
    T,P = S.shape
    
    max_iteration = P * 10
    R = float(nonzero * P)

    # Normalize the data.
    S -= S.mean(axis = 0)
    S /= da.vnorm(S, axis = 0)

    # Generate the atom vectors.
    u_old = da.zeros(T, chunks=(1,))
    u_new = da.zeros(T,chunks=(1,))
    v = da.zeros(P,chunks=(1,))
    Z = np.zeros((atoms,P),dtype = np.float)
    D = np.zeros((atoms, T),dtype = np.float)
    idxs_n = da.zeros(int(R),chunks=(1,))
    epsilon *= epsilon
    
    for m in range(atoms):
        print ('m: '+str(m))
        it = 0
        u_old = da.random.random(T,chunks=(1,))
        
        u_old -= u_old.mean()
        u_old /= da.vnorm(u_old, axis = 0)
        while True:
            print ('it: '+str(it))
            v = da.dot(u_old,S)
    
            # Zero out all elements of v NOT in the top-R. This is how
            # sparsity in the final results is explicitly enforced.
            max_values = op_selectTopR(v, int(R))

            '''
            UGLY HACK!!

            Dask arrays are immutable. So, couldn't find a 
            way to zero out all elements of v not in topR or 
            the other way round.
            The only option we could think of was to
            extract the max value and match it with v. All other
            indices are initialzed to 0 but the matched one.  
            This will work only in cases with len(max_values) = 1
            as np.where cannot accomodate a list in where condition.
            Also, dask doesn't seem to have any function which
            can get indices of array.
            '''
            max_values = max_values.flatten()
            max_values = max_values[0].compute()
            v  = da.where(v == max_values,v,0)
            v = v.compute()
            v_np  = np.array(v)
            idxs_n = np.argwhere(v_np == max_values).flatten()   
            
            u_new = da.dot(S[:, idxs_n], v[idxs_n])
            u_new /= da.vnorm(u_new, axis = 0)
            diff = da.vnorm(u_old - u_new)
            diff = diff.compute()
            
            if (diff < epsilon):
                break
            if (it > max_iteration):
                print('WARNING: Max iteration reached; result may be unstable!\n')
                break
                
            
            # Copying the new vector on old one
            u_old = u_new.compute()
            it += 1
            
      
        S = op_getResidual(S, u_new, v, idxs_n)
        
        Z[m, :] = v
        D[m, :] = u_new
    
        
    # All done!
    return [D, Z]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Python Dictionary Learning',
        add_help = 'How to use', prog = 'python R1DL.py <args>')

    # Input arguments.
    parser.add_argument("-i", "--input", required = True,
        help = "Input filename containing matrix S.")
    parser.add_argument("-r", "--pnonzero", type = float, required = True,
        help = "Percentage of non-zero elements.")
    parser.add_argument("-m", "--mDicatom", type = int, required = True,
        help = "Number of the dictionary atoms.")
    parser.add_argument("-e", "--epsilon", type = float, required = True,
        help = "The value of epsilon.")

    # Output arguments.
    parser.add_argument("-d", "--dictionary", required = True,
        help = "Dictionary (D) output file.")
    parser.add_argument("-z", "--zmatrix", required = True,
        help = "Loading matrix (Z) output file.")

    args = vars(parser.parse_args())

    # Parse out the command-line arguments.
    M = args['mDicatom']
    R = args['pnonzero']
    epsilon = args['epsilon']
    file_s = args['input']
    file_D = args['dictionary']
    file_Z = args['zmatrix']

    # Read the inputs and generate variables to pass to R1DL.
    S = np.loadtxt(file_s)
    Sd = da.from_array(S, chunks=(1,1))
    
    D, Z = r1dl(Sd, R, M, epsilon)
    
    # Write the output to files.
    np.savetxt(file_D, D, fmt = '%.5lf\t')
    np.savetxt(file_Z, Z, fmt = '%.5lf\t')
    
