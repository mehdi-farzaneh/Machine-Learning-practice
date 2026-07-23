import numpy as np
import sys

def fmincg(f, X, options=None, *args):
    """
    Minimize a continuous differentiable multivariate function. Starting point
    is given by "X" (D by 1), and the function "f" must return a function value 
    and a vector of partial derivatives.
    
    Usage: X, fX, i = fmincg(f, X, options, P1, P2, P3, P4, P5)
    
    Parameters:
    -----------
    f : function
        Function to minimize. Should return (function_value, gradient)
    X : array-like
        Starting point (1D array)
    options : dict, optional
        Dictionary containing 'MaxIter' key for maximum iterations
    *args : variable args
        Additional arguments to pass to function f
    
    Returns:
    --------
    X : ndarray
        Optimized parameters
    fX : list
        Function values at each iteration
    i : int
        Number of iterations performed
    """
    
    # Read options
    if options is not None and 'MaxIter' in options:
        length = options['MaxIter']
    else:
        length = 100
    
    # Constants for line searches
    RHO = 0.01      # RHO and SIG are constants in the Wolfe-Powell conditions
    SIG = 0.5
    INT = 0.1       # Don't reevaluate within 0.1 of the limit
    EXT = 3.0       # Extrapolate maximum 3 times the current bracket
    MAX = 20        # Max 20 function evaluations per line search
    RATIO = 100     # Maximum allowed slope ratio
    
    # Handle length as [maxiter, reduction] pair
    if isinstance(length, (list, tuple, np.ndarray)):
        if len(length) == 2:
            red = length[1]
            length = length[0]
        else:
            red = 1
    else:
        red = 1
    
    # Initialize
    i = 0
    ls_failed = 0
    fX = []
    X = np.array(X, dtype=float).flatten()
    
    # Get initial function value and gradient
    f1, df1 = f(X, *args)
    df1 = np.array(df1, dtype=float).flatten()
    
    i = i + (1 if length < 0 else 0)
    s = -df1  # Search direction is steepest descent
    d1 = -np.dot(s, s)  # Slope
    z1 = red / (1 - d1)  # Initial step
    
    # Main optimization loop
    while i < abs(length):
        i = i + (1 if length > 0 else 0)
        
        X0 = X.copy()
        f0 = f1
        df0 = df1.copy()
        
        X = X + z1 * s  # Begin line search
        f2, df2 = f(X, *args)
        f2 = float(f2)
        df2 = np.array(df2, dtype=float).flatten()
        
        i = i + (1 if length < 0 else 0)
        d2 = np.dot(df2, s)
        f3 = f1
        d3 = d1
        z3 = -z1
        
        M = MAX if length > 0 else min(MAX, -length - i)
        success = 0
        limit = -1
        
        # Line search loop
        while True:
            # Inner loop for bracketing
            while (f2 > f1 + z1 * RHO * d1 or d2 > -SIG * d1) and M > 0:
                limit = z1
                
                if f2 > f1:
                    # Quadratic fit
                    z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3)
                else:
                    # Cubic fit
                    A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3)
                    B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2)
                    z2 = (np.sqrt(B * B - A * d2 * z3 * z3) - B) / A
                
                # Handle numerical errors
                if np.isnan(z2) or np.isinf(z2):
                    z2 = z3 / 2
                
                z2 = max(min(z2, INT * z3), (1 - INT) * z3)
                z1 = z1 + z2
                X = X + z2 * s
                f2, df2 = f(X, *args)
                f2 = float(f2)
                df2 = np.array(df2, dtype=float).flatten()
                M = M - 1
                i = i + (1 if length < 0 else 0)
                d2 = np.dot(df2, s)
                z3 = z3 - z2
            
            # Check line search result
            if f2 > f1 + z1 * RHO * d1 or d2 > -SIG * d1:
                break  # Failure
            elif d2 > SIG * d1:
                success = 1
                break  # Success
            elif M == 0:
                break  # Failure
            
            # Make cubic extrapolation
            A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3)
            B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2)
            discriminant = B * B - A * d2 * z3 * z3
            
            if discriminant >= 0:
                z2 = -d2 * z3 * z3 / (B + np.sqrt(discriminant))
            else:
                z2 = np.nan
            
            # Handle numerical problems
            if np.isnan(z2) or np.isinf(z2) or z2 < 0:
                if limit < -0.5:
                    z2 = z1 * (EXT - 1)
                else:
                    z2 = (limit - z1) / 2
            elif limit > -0.5 and z2 + z1 > limit:
                z2 = (limit - z1) / 2
            elif limit < -0.5 and z2 + z1 > z1 * EXT:
                z2 = z1 * (EXT - 1.0)
            elif z2 < -z3 * INT:
                z2 = -z3 * INT
            elif limit > -0.5 and z2 < (limit - z1) * (1.0 - INT):
                z2 = (limit - z1) * (1.0 - INT)
            
            f3 = f2
            d3 = d2
            z3 = -z2
            z1 = z1 + z2
            X = X + z2 * s
            f2, df2 = f(X, *args)
            f2 = float(f2)
            df2 = np.array(df2, dtype=float).flatten()
            M = M - 1
            i = i + (1 if length < 0 else 0)
            d2 = np.dot(df2, s)
        
        # End of line search
        
        if success:  # Line search succeeded
            f1 = f2
            fX.append(f1)
            print(f'Iteration {i:4d} | Cost: {f1:4.6e}', end='\r')
            sys.stdout.flush()
            
            # Polack-Ribiere direction
            denom = np.dot(df1, df1)
            if denom != 0:
                s = ((np.dot(df2, df2) - np.dot(df1, df2)) / denom) * s - df2
            else:
                s = -df2
            
            df1 = df2.copy()
            d2 = np.dot(df1, s)
            
            if d2 > 0:  # New slope must be negative
                s = -df1
                d2 = -np.dot(s, s)
            
            z1 = z1 * min(RATIO, d1 / (d2 - np.finfo(float).tiny))
            d1 = d2
            ls_failed = 0
        else:
            X = X0.copy()
            f1 = f0
            df1 = df0.copy()
            
            if ls_failed or i > abs(length):
                break
            
            df1 = df2.copy()
            s = -df1
            d1 = -np.dot(s, s)
            z1 = 1 / (1 - d1)
            ls_failed = 1
    
    print()
    return X, fX, i
