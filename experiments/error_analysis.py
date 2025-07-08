import jax
import jax.numpy as jnp

import signax
from polysigkernel.sigkernel import SigKernel as SigKernel_polynomial
from sigkerax.sigkernel import SigKernel as SigKernel_sigkerax

import pandas as pd
import timeit
from tqdm import tqdm
import argparse
import time
import random 

def mean_absolute_error(ker1, ker2):
    return jnp.mean(jnp.abs(ker1 - ker2))

def mean_absolute_percentage_error(ker1, ker2):
    return jnp.mean(jnp.abs((ker1-ker2) / ker2))

def root_mean_squared_error(ker1, ker2):
    return jnp.linalg.norm(ker1 - ker2)

def get_truncated_signature_level(dim):
    if dim == 2:
        return 21
    elif dim == 3:
        return 16
    elif dim == 4:
        return 10

def get_key():
    return jax.random.PRNGKey(random.randint(0, 2 ** 31 - 1))


def signax_kernel(X, Y, level : int):
    sigs_X = signax.signature(X, level)
    sigs_Y = signax.signature(Y, level)
    return 1. + jnp.einsum("xa,ya -> xy", sigs_X, sigs_Y)


if __name__ == '__main__':

    #  change arg parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='error_results2', help='Name of the file to save the results.')

    parser.add_argument('--sigkerax', help='True to do analysis on sigkerax, False (default) otherwise.', action='store_true')
    parser.add_argument('--solvers', type=str, nargs='+', default=['monomial_approx', 'monomial_interp'], help='Solvers to use for the polynomial signature kernel.')
    parser.add_argument('--data', type=str, default='bm', choices=['bm', 'smooth'], help='Data to use for the analysis.')

    parser.add_argument('--compute_time', help='True (default) to compute the time taken for the analysis, False  otherwise.', action='store_false')

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the data.')
    
    parser.add_argument('--number', type=int, default=4, help='Number fo runs in a repeat loop')
    parser.add_argument('--repeat', type=int, default=4, help='Number of repeat loops')
    parser.add_argument('--debug', help='One param per sequence', action='store_true')
    parser.add_argument('--save', help='Saves csv file', action='store_false')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'], help='Device to run the code.')
    parser.add_argument('--dtype', type=str, default='default', choices=['default', 'float32', 'float64'], 
                        help='Data type for the data. Default is float32 for gpu and float64 for cpu.')

    args = parser.parse_args()

    if isinstance(args.solvers, str):
        args.solvers = [args.solvers]


    def timer_func(func):
        timer = timeit.Timer(lambda : func())
        results = timer.repeat(number=args.number, repeat=args.repeat)
        elapsed_time = sum(results) / args.number / args.repeat
        return elapsed_time

    _columns = [
                'model',
                'device',
                'dtype',
                'batch_size', 
                'length',  
                'dim', 
                'level', # or 'refinement_factor
                'benchmark_level', 
                'mae',
                'mape',
                'rmse',
                'time'
                ]
    
    _lengths = [10,50,100,200]
    _dims = [2,3] 
    _orders = [i for i in range(2, 18)]
    _refinement_factors = [i for i in range(1,18)]

    if args.debug:
        _lengths = [200]
        _dims = [2] 
        _orders = [18]
        _refinement_factors = [2]

    ############################################################################################################
    # Set the device and data type
    ############################################################################################################

    if args.device == 'cpu':
        if args.dtype == 'default' or args.dtype == 'float64':
            dtype = jnp.float64
        else:
            dtype = jnp.float32
    
    elif args.device == 'gpu':
        if args.dtype == 'default' or args.dtype == 'float32':
            dtype = jnp.float32
        else:
            dtype = jnp.float64

    if dtype == jnp.float64:
        jax.config.update('jax_enable_x64', True) # Enable 64-bit precision

    dtype_str = 'float32' if dtype == jnp.float32 else 'float64'

    ############################################################################################################
    # Run the analysis
    ############################################################################################################

    df = pd.DataFrame(columns = _columns)
    
    lengths = tqdm(_lengths, position=0, leave=True)
    for length in lengths:

        dims = tqdm(_dims, position=1, leave=False)
        for dim in dims:
            
            # Generate data
            key1 = jax.random.PRNGKey(280201)
            key2 = jax.random.PRNGKey(280202)

            if args.data == 'bm':
                X = 1e-1 * jax.random.normal(key1, shape=(args.batch_size, length, dim), dtype=dtype).cumsum(axis=1)
                Y = 1e-1 * jax.random.normal(key2, shape=(args.batch_size, length, dim), dtype=dtype).cumsum(axis=1)

            elif args.data == 'smooth':
                X = 5e-1 * jnp.sin(jax.random.normal(key1, shape=(args.batch_size, length, dim), dtype=dtype))
                Y = 5e-1 * jnp.cos(jax.random.normal(key2, shape=(args.batch_size, length, dim), dtype=dtype))


            if dtype == jnp.float32:

                jax.config.update('jax_enable_x64', True) # Enable 64-bit precision

                if args.data == 'bm':
                    X64 = 1e-1 * jax.random.normal(key1, shape=(args.batch_size, length, dim), dtype=jnp.float64).cumsum(axis=1)
                    Y64 = 1e-1 * jax.random.normal(key2, shape=(args.batch_size, length, dim), dtype=jnp.float64).cumsum(axis=1)
                
                elif args.data == 'smooth':
                    X64 = 5e-1 * jnp.sin(jax.random.normal(key1, shape=(args.batch_size, length, dim), dtype=jnp.float64))
                    Y64 = 5e-1 * jnp.cos(jax.random.normal(key2, shape=(args.batch_size, length, dim), dtype=jnp.float64))

                # Compute trucated signature kernel
                level = get_truncated_signature_level(dim)
                signax_ker = signax_kernel(X64, Y64, level).astype(dtype)

                jax.config.update('jax_enable_x64', False) 

            else:
                # Compute trucated signature kernel
                level = get_truncated_signature_level(dim)
                signax_ker = signax_kernel(X, Y, level)

            _solvers = tqdm(args.solvers, position=2, leave=False)
            for solver in _solvers:

                # Compute polynomial signature kernel
                orders = tqdm(_orders, position=2, leave=False)
                for order in orders:

                    sigkernel = SigKernel_polynomial(order=order, static_kernel='linear', add_time=False, solver=solver)
                    sigkernel_matrix = sigkernel.kernel_matrix(X, Y)

                    polysigker_mae = mean_absolute_error(sigkernel_matrix, signax_ker)
                    polysigker_mape = mean_absolute_percentage_error(sigkernel_matrix, signax_ker)
                    polysigker_rmse = root_mean_squared_error(sigkernel_matrix, signax_ker)

                    if args.compute_time:

                        sk = SigKernel_polynomial(order=order, static_kernel='linear', add_time=False, solver=solver)
                        sigkernel_poly = lambda : sk.kernel_matrix(X, Y)
                        time_elapsed = timer_func(sigkernel_poly)

                        df.loc[len(df.index)] = [solver, args.device, dtype_str, args.batch_size, length, dim, order, level, 
                                                polysigker_mae, polysigker_mape, polysigker_rmse, time_elapsed]
                    
                    else:    
                        df.loc[len(df.index)] = [solver, args.device, dtype_str, args.batch_size, length, dim, order, level, 
                                                polysigker_mae, polysigker_mape, polysigker_rmse, 0]
                        
                    time.sleep(0.5)


            if args.sigkerax:
                
                refinement_factors = tqdm(_refinement_factors, position=2, leave=False)
                
                for refinement_factor in refinement_factors:

                    # Compute sigkerax kernel
                    sigkerax_obj = SigKernel_sigkerax(refinement_factor=refinement_factor, static_kernel_kind='linear', add_time=False)
                    sigkerax_matrix = sigkerax_obj.kernel_matrix(X, Y)[...,0]

                    sigkerax_mae  = mean_absolute_error(sigkerax_matrix, signax_ker)
                    sigkerax_mape = mean_absolute_percentage_error(sigkerax_matrix, signax_ker)
                    sigkerax_rmse = root_mean_squared_error(sigkerax_matrix, signax_ker)

                    if args.compute_time:

                        sk = SigKernel_sigkerax(refinement_factor=refinement_factor, static_kernel_kind='linear', add_time=False)
                        sigkernel_sigkerax = lambda : sk.kernel_matrix(X, Y)[...,0]
                        time_elapsed = timer_func(sigkernel_sigkerax)
                
                        df.loc[len(df.index)] = ['sigkerax', args.device, dtype_str, args.batch_size, length, dim, refinement_factor, level,
                                                 sigkerax_mae, sigkerax_mape, sigkerax_rmse, time_elapsed]

                    else:
                        df.loc[len(df.index)] = ['sigkerax', args.device, dtype_str, args.batch_size, length, dim, refinement_factor, level,
                                                 sigkerax_mae, sigkerax_mape, sigkerax_rmse, 0]

                    time.sleep(0.5)

            if args.save:

                df.to_csv('../results/' + args.filename + '_' + args.device + '.csv', index=False)


    print("DONE")





