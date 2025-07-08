import jax
import jax.numpy as jnp

from polysigkernel.sigkernel import SigKernel as SigKernel_polynomial
from sigkerax.sigkernel import SigKernel as SigKernel_sigkerax

import pandas as pd
from tqdm import tqdm
import argparse
import timeit


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='time_results', help='Name of the file to save the results.')
    
    parser.add_argument('--length', help='Run the length analysis - False default', action='store_false')
    parser.add_argument('--dim', help='Run the dim analysis - False default', action='store_false')
    parser.add_argument('--order', help='Run the order analysis - False default', action='store_false')

    parser.add_argument('--polysigker', help='True to do analysis on sigkerax, False (default) otherwise.', action='store_false')
    parser.add_argument('--sigkerax', help='True to do analysis on sigkerax, False (default) otherwise.', action='store_true')
    parser.add_argument('--solvers', type=str, nargs='+', default=['monomial_approx'], help='Solvers to use for the polynomial signature kernel.')
    
    parser.add_argument('--large', action='store_true', help='increases arrays of orders')    
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for the data.')
    parser.add_argument('--def_length', type=int, default=32, help='Default length of the data.')
    parser.add_argument('--def_dim', type=int, default=8, help='Default dimension of the data.')

    parser.add_argument('--number', type=int, default=6, help='Number fo runs in a repeat loop')
    parser.add_argument('--repeat', type=int, default=6, help='Number of repeat loops')

    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'], help='Device to run the code.')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'], 
                        help='Data type for the data. Default is float32 for gpu and float64 for cpu.')

    args = parser.parse_args()

    if isinstance(args.solvers, str):
        args.solvers = [args.solvers]

    def timer_func(func):
        timer = timeit.Timer(lambda : func())
        results = timer.repeat(number=args.number, repeat=args.repeat)
        #elapsed_time = min(results) / args.number
        elapsed_time = sum(results) / args.number / args.repeat
        return elapsed_time

    _columns = [
                'analysis',
                'model',
                'device',
                'dtype',
                'batch_size', 
                'length',  
                'dim', 
                'param', # 'dim' or 'refinement_factor
                'time'
                ]
    
    # get default arguments 
    
    _lengths = list(range(100, 501, 25))
    _dims = list(range(50, 2001, 50))
    _orders = [2,3,4,5]
    _refinement_factors = [1,2,3,4]

    default_length = args.def_length
    default_dim = args.def_dim

    dtype_str = args.dtype
    dtype = jnp.float64 if args.dtype == 'float64' else jnp.float32

    if dtype == jnp.float64:
        jax.config.update('jax_enable_x64', True) # Enable 64-bit precision

    df = pd.DataFrame(columns = _columns)

    #==================================================================================
    # Length analysis
    #==================================================================================
    if args.length:

        print('Starting length analysis ...')

        lengths = tqdm(_lengths, position=0, leave=True)
        for length in lengths:

            # Generate data
            X = 1e-1 * jax.random.normal(jax.random.PRNGKey(0), shape=(args.batch_size, length, default_dim), dtype=dtype).cumsum(axis=1)
            Y = 1e-1 * jax.random.normal(jax.random.PRNGKey(1), shape=(args.batch_size, length, default_dim), dtype=dtype).cumsum(axis=1)

            if args.polysigker:
                
                for solver in args.solvers:

                    # Compute polynomial signature kernel
                    orders = tqdm(_orders, position=1, leave=False)
                    for order in orders:

                        sigkernel_poly = lambda : SigKernel_polynomial(order=order, static_kernel='linear', solver=solver,
                                                                    add_time=False).kernel_matrix(X, Y)
                        time_elapsed = timer_func(sigkernel_poly)

                        df.loc[len(df.index)] = ['length', solver, args.device, dtype_str, args.batch_size, length, default_dim, order, time_elapsed]

            if args.sigkerax:
                
                # Compute sigkerax kernel
                sigkernel_sigkerax = lambda : SigKernel_sigkerax(refinement_factor=1, static_kernel_kind='linear', 
                                                                add_time=False).kernel_matrix(X, Y)[...,0]
                
                time_elapsed = timer_func(sigkernel_sigkerax)

                df.loc[len(df.index)] = ['length', 'sigkerax', args.device, dtype_str, args.batch_size, length, default_dim, 1, time_elapsed]


    #==================================================================================
    # Dimension analysis
    #==================================================================================

    if args.dim:
         
        print('Starting dimension analysis ...')

        dims = tqdm(_dims, position=0, leave=True)
        for dim in dims:
            
            # Generate data
            X = 1e-1 * jax.random.normal(jax.random.PRNGKey(0), shape=(args.batch_size, default_length, dim), dtype=dtype).cumsum(axis=1)
            Y = 1e-1 * jax.random.normal(jax.random.PRNGKey(1), shape=(args.batch_size, default_length, dim), dtype=dtype).cumsum(axis=1)

            if args.polysigker:

                for solver in args.solvers:
                    
                    # Compute polynomial signature kernel
                    orders = tqdm(_orders, position=1, leave=False)
                    for order in orders:

                        sigkernel_poly = lambda : SigKernel_polynomial(order=order, static_kernel='linear', solver=solver,
                                                                    add_time=False).kernel_matrix(X, Y)

                        time_elapsed = timer_func(sigkernel_poly)

                        df.loc[len(df.index)] = ['dim', solver, args.device, dtype_str, args.batch_size, default_length, dim, order, time_elapsed]

            if args.sigkerax:
                    
                # Compute sigkerax kernel with sigkerax
                refinement_factors = tqdm(_refinement_factors, position=1, leave=False)
                for refinement_factor in refinement_factors:
                
                    sigkernel_sigkerax = lambda: SigKernel_sigkerax(refinement_factor=refinement_factor, static_kernel_kind='linear', 
                                                                    add_time=False).kernel_matrix(X, Y)[...,0]

                    time_elapsed = timer_func(sigkernel_sigkerax)

                    df.loc[len(df.index)] = ['dim', 'sigkerax', args.device, dtype_str, args.batch_size, default_length, dim, refinement_factor, time_elapsed]

    
    #==================================================================================
    # Order analysis
    #==================================================================================
        
    if args.order:

        print('Starting order analysis ...')
            
        # Generate data
        X = 1e-1 * jax.random.normal(jax.random.PRNGKey(0), shape=(args.batch_size, default_length, default_dim), dtype=dtype).cumsum(axis=1)
        Y = 1e-1 * jax.random.normal(jax.random.PRNGKey(1), shape=(args.batch_size, default_length, default_dim), dtype=dtype).cumsum(axis=1)

        _orders1 = [i for i in range(2, 20)]
        _refinement_factors1 = [i for i in range(2, 20)]

        if args.large:
            _orders1 = [i for i in range(2, 40)]	
        
        if args.polysigker:

            for solver in args.solvers:
                
                # Compute polynomial signature kernel with monomial_approx
                orders = tqdm(_orders1, position=0, leave=True)
                for order in orders:

                    sigkernel_poly = lambda : SigKernel_polynomial(order=order, static_kernel='linear', solver=solver,
                                                                add_time=False).kernel_matrix(X, Y)

                    time_elapsed = timer_func(sigkernel_poly)

                    df.loc[len(df.index)] = ['order', solver, args.device, dtype_str, args.batch_size, default_length, default_dim, order, time_elapsed]
		
            df.to_csv('results/' + args.filename + '.csv', index=False)

        if args.sigkerax:
                    
            # Compute sigkerax kernel with sigkerax
            refinement_factors = tqdm(_refinement_factors1, position=0, leave=False)
            for refinement_factor in refinement_factors:
            
                sigkernel_sigkerax = lambda: SigKernel_sigkerax(refinement_factor=refinement_factor, static_kernel_kind='linear', 
                                                                add_time=False).kernel_matrix(X, Y)[...,0]

                time_elapsed = timer_func(sigkernel_sigkerax)

                df.loc[len(df.index)] = ['order', 'sigkerax', args.device, dtype_str, args.batch_size, default_length, default_dim, refinement_factor, time_elapsed]

            df.to_csv('results/' + args.filename + '.csv', index=False)	

    df.to_csv('results/' + args.filename + '.csv', index=False)
    
