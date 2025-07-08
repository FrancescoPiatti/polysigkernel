import jax
import jax.numpy as jnp
from functools import partial

from .monomial_approximation_solver import MonomialApproximationSolver
from .monomial_interpolation_solver import MonomialInterpolationSolver
from .utils import add_time_fn, interpolate_fn
from .config import _SOLVERS, _KERNELS, _INTERPOLATIONS
from .checks import _check_positive_integer, _check_positive_value


class SigKernel:
    """
    Signature Kernel class for computing kernel matrices, distances, scoring rules, 
    and MMD-based measures on time-series (path) data. 
    Implementation from https://arxiv.org/abs/2502.08470.
    """
    def __init__(self, 
                 order : int = 5,
                 static_kernel: str = 'linear',
                 solver : str = 'monomial_approx', 
                 refinement_factor : int = 1,
                 scales: jnp.ndarray = jnp.array([1e0]),
                 s0 : float = 0., 
                 t0 : float = 0., 
                 S : float = 1., 
                 T : float = 1., 
                 add_time : bool = False,
                 interpolation : str = "linear",
                 multi_gpu : bool = False):
        
        _check_positive_integer(order, "order")
        _check_positive_integer(refinement_factor, "refinement_factor")
        _check_positive_value(scales[0], "scales[0]")
        
        if static_kernel not in _KERNELS:
            raise ValueError("Static kernel not implemented.")
        
        if solver not in _SOLVERS:  
            raise ValueError("Solver not implemented.")
        
        if interpolation not in _INTERPOLATIONS:
            raise ValueError("Interpolation not implemented.")
        
        if solver == "monomial_approx":
            self.solver = MonomialApproximationSolver
        elif solver == "monomial_interp":
            self.solver = MonomialInterpolationSolver 

        self.static_kernel = static_kernel
        self.order = order
        self.refinement_factor = refinement_factor
        self.interpolation = interpolation
        self.scales = scales
        self.multi_gpu = multi_gpu

        # LEGACY - TO CHANGE
        self.scale = self.scales[0] 

        self.add_time = add_time
        self.s0 = s0
        self.t0 = t0
        self.S = S
        self.T = T


    @partial(jax.jit, static_argnums=(0,3,4))
    def kernel_matrix(self, 
                      X: jnp.ndarray, 
                      Y: jnp.ndarray, 
                      max_batch : int = None,
                      sym : bool = False) -> jnp.ndarray:
        """
        Compute the signature kernel matrix between sets of paths X and Y.

        If max_batch is specified and the batch dimension exceeds that limit,
        the computation is split recursively to avoid memory issues.

        Args:
            X (jax.numpy.ndarray): Shape (batch_X, length_X, dim).
            Y (jax.numpy.ndarray): Shape (batch_Y, length_Y, dim).
            max_batch (int, optional): Maximum allowed batch size for a single 
                                       kernel computation. Defaults to None 
                                       (i.e., no splitting).
            sym (bool, optional): If True, the result is treated as symmetric, and 
                                  sub-blocks are computed accordingly. Defaults to False.

        Returns:
            jax.numpy.ndarray: Kernel matrix of shape (batch_X, batch_Y).
        """
        
        if self.refinement_factor > 1:
            # Refine X and Y via interpolation
            X = interpolate_fn(X, t_min=self.s0, t_max=self.S, refinement_factor=self.refinement_factor, 
                               kind=self.interpolation)
            Y = interpolate_fn(Y, t_min=self.t0, t_max=self.T, refinement_factor=self.refinement_factor, 
                               kind=self.interpolation)

        # Optionally add time as an extra channel
        if self.add_time:
            X = add_time_fn(X, t_min=self.s0, t_max=self.S)
            Y = add_time_fn(Y, t_min=self.t0, t_max=self.T)

        batch_X, batch_Y = X.shape[0], Y.shape[0]
        
        # If no splitting is necessary (or no max_batch is provided):
        if (max_batch is None) or (batch_X <= max_batch and batch_Y <= max_batch):
            return self.solver(static_ker=self.static_kernel, scale=self.scale, 
                               order=self.order).solve(X, Y, sym, self.multi_gpu)
        
        # Case 1: X small enough, Y large
        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y/2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            K1 = self.kernel_matrix(X, Y1, max_batch, False)
            K2 = self.kernel_matrix(X, Y2, max_batch, False)
            return jnp.concatenate((K1, K2), axis=1)
        
        # Case 2: X large, Y small enough
        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            K1 = self.kernel_matrix(X1, Y, max_batch, False)
            K2 = self.kernel_matrix(X2, Y, max_batch, False)
            return jnp.concatenate((K1, K2), axis=0)
        
        # Case 3: Both X, Y large
        else:
            cutoff_X, cutoff_Y = int(batch_X/2), int(batch_Y/2)
            X1, X2 = X[:cutoff_X], X[cutoff_X:]
            Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]

            # Compute sub-blocks
            K11 = self.kernel_matrix(X1, Y1, max_batch, sym)
            K12 = self.kernel_matrix(X1, Y2, max_batch, False)
            K22 = self.kernel_matrix(X2, Y2, max_batch, sym)

            if sym:
                K21 = K12.swapaxes(0, 1)
            else:
                K21 = self.kernel_matrix(X2, Y1, max_batch, False)

            K_top = jnp.concatenate((K11, K12), axis=1)
            K_bottom = jnp.concatenate((K21, K22), axis=1)
            K = jnp.concatenate((K_top, K_bottom), axis=0)
            return K
        
    ########################################################################################

    def compute_distance(self, X, Y, max_batch=100):
        """
        Input:
            - X: jnp array of shape (batch, length_X, dim),
            - Y: jnp array of shape (batch, length_Y, dim)
        Output: 
            - vector ||S(X^i)_T - S(Y^i)_T||^2 of shape (batch,)
        """

        K_XX = self.kernel_matrix(X, X, max_batch=max_batch)
        K_YY = self.kernel_matrix(Y, Y, max_batch=max_batch)
        K_XY = self.kernel_matrix(X, Y, max_batch=max_batch)

        return jnp.mean(K_XX) + jnp.mean(K_YY) - 2.*jnp.mean(K_XY)

    def compute_scoring_rule(self, X, y, max_batch=100):
        """
        Input:
            - X: jnp array of shape (batch, length_X, dim),
            - y: jnp array of shape (1, length_Y, dim)
        Output:
            - signature kernel scoring rule S(X,y) = E[k(X,X)] - 2E[k(X,y)]
        """

        K_XX = self.compute_Gram(X, X, sym=True, max_batch=max_batch)
        K_Xy = self.compute_Gram(X, y, sym=False, max_batch=max_batch)

        K_XX_m = (jnp.sum(K_XX) - jnp.sum(jnp.diag(K_XX))) / (K_XX.shape[0] * (K_XX.shape[0] - 1.))

        return K_XX_m - 2. * jnp.mean(K_Xy)

    def compute_expected_scoring_rule(self, X, Y, max_batch=100):
        """
        Input:
            - X: jnp array of shape (batch_X, length_X, dim),
            - Y: jnp array of shape (batch_Y, length_Y, dim)
        Output:
            - signature kernel expected scoring rule S(X,Y) = E_Y[S(X,y)]
        """

        assert not Y.requires_grad, "the second input should not require grad"

        K_XX = self.kernel_matrix(X, X, sym=True, max_batch=max_batch)
        K_XY = self.kernel_matrix(X, Y, sym=False, max_batch=max_batch)

        K_XX_m = (jnp.sum(K_XX) - jnp.sum(jnp.diag(K_XX))) / (K_XX.shape[0] * (K_XX.shape[0] - 1.))

        return K_XX_m - 2. * jnp.mean(K_XY)

    def compute_mmd(self, X, Y, max_batch=100):
        """
        Input:
            - X: jnp array of shape (batch_X, length_X, dim),
            - Y: jnp array of shape (batch_Y, length_Y, dim)
        Output: 
            - scalar: MMD signature distance between samples X and samples Y
        """

        K_XX = self.kernel_matrix(X, X, sym=True, max_batch=max_batch)
        K_YY = self.kernel_matrix(Y, Y, sym=True, max_batch=max_batch)
        K_XY = self.kernel_matrix(X, Y, sym=False, max_batch=max_batch)

        K_XX_m = (jnp.sum(K_XX) - jnp.sum(jnp.diag(K_XX))) / (K_XX.shape[0] * (K_XX.shape[0] - 1.))
        K_YY_m = (jnp.sum(K_YY) - jnp.sum(jnp.diag(K_YY))) / (K_YY.shape[0] * (K_YY.shape[0] - 1.))

        return K_XX_m + K_YY_m - 2. * jnp.mean(K_XY)



########################################################################################
# Hypothesis test functionality
########################################################################################


def c_alpha(m, alpha):
    return 4. * jnp.sqrt(-jnp.log(alpha) / m)

def hypothesis_test(y_pred : jnp.array, 
                    y_test : jnp.array, 
                    static_kernel : str = 'linear', 
                    confidence_level : float = 0.99, 
                    **kwargs):
    """
    Statistical test based on MMD distance to determine if
    two sets of paths come from the same distribution.
    """

    order = kwargs.get('order', 3)
    max_batch = kwargs.get('max_batch', 100)
    solver = kwargs.get('solver', 'monomial_approx')
    refinement_factor = kwargs.get('refinement_factor', 1)
    interpolate_fn = kwargs.get('interpolate_fn', 'linear')
    scales = kwargs.get('scales', jnp.array([1e0]))

    k_sig = SigKernel(order, static_kernel, solver=solver, refinement_factor=refinement_factor,
                      scales=scales, interpolation=interpolate_fn)

    m = max(y_pred.shape[0], y_test.shape[0])

    TU = k_sig.compute_mmd(y_pred,y_test, max_batch=max_batch)

    c = jnp.array(c_alpha(m, confidence_level), dtype=y_pred.dtype)

    if TU > c:
        print(f'Hypothesis rejected: distribution are not equal with {confidence_level*100}% confidence')
    else:
        print(f'Hypothesis accepted: distribution are equal with {confidence_level*100}% confidence')

# ===========================================================================================================



