import numpy as np
from scipy.optimize import fmin_cg
from scipy.linalg import expm
from scipy import integrate
from functools import lru_cache, wraps

def _totuple(a):
    try:
        return tuple(_totuple(i) for i in a)
    except TypeError:
        return a.item()

def np_cache(function):
    @lru_cache()
    def cached_wrapper(*hashable_args):
        args = (np.array(arg) for arg in hashable_args)
        return function(*(args))

    @wraps(function)
    def wrapper(*array_args):
        return cached_wrapper(*(_totuple(arg) for arg in array_args))

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


def gradient_descent(f, x0, grad_f,
                     args = (), gtol=1e-05, maxiter = None,
                     step_size = 1, step_choice = None,
                     callback = lambda x: None,
                     full_output = False,):
#TO DO: norm, disp, approximate gradient, other exit conditions?
#TO DO: line search for choosing stepsize

    if(maxiter is None):
        maxiter = 200*len(x0)
    warnflag = None
    x = x0
    
    for i in range(maxiter):
        
        callback(x)
        
        gradvec = grad_f(x, *args)
        
        if(np.max(np.abs(gradvec)) < gtol):
            warnflag = 0
            break
        
        if step_choice == 'BB':
            #Barzilai–Borwein method
            if i == 0:
                pass
            else:
                step_size = np.abs(np.dot(x-x_prev, gradvec - gradvec_prev))/np.dot(gradvec - gradvec_prev, gradvec - gradvec_prev)
            x_prev = x
            gradvec_prev = gradvec
            
        x = x - step_size*gradvec
    
    if warnflag is None:
        warnflag = 1
    fopt = f(x, *args)
    
    message = {
        0:'Optimization terminated successfully',
        1:'Warning: Desired precision not necessarily achieved. Max number of iterations reached.'
    }[warnflag]
    print(message)
    print(f'''
            Current function value: {fopt}
            Iterations: {i}''')
    if full_output:
        return x, fopt, i, i, 1
    return x
        

def parameter_error(theta, theta0, data_sim, data_exp, weights, sigmas):
    '''
    Cost function as in Eq. (4)
    all inputs as numpy arrays,
    theta, theta0, weights of matching dimensions,
    data_sim, data_exp, sigmas of matching dimensions.
    '''
    
    term1 = np.sum((theta-theta0)**2/weights)/2
    term2 = np.sum((data_sim-data_exp)**2/sigmas)/2   
    return term1+term2

def find_parameters(data_exp, theta_init, weights, sigmas, sim_fun, grad_fun, optimizer, optimizer_kwargs = {}):
    
    target_fun = lambda theta: parameter_error(theta, theta_init, sim_fun(theta), data_exp, weights, sigmas)
    callback_fun = lambda x: clear_all_caches()
     
    if optimizer == 'scipy_cg':
        return fmin_cg(
            target_fun,
            theta_init,
            grad_fun,
            **optimizer_kwargs,
            callback = callback_fun
        )
    if optimizer == 'gd':
        return gradient_descent(
            target_fun,
            theta_init,
            grad_fun,
            **optimizer_kwargs,
            callback = callback_fun
        )

def hamiltonian_learning(hamiltonian_operators, exp_settings, data_exp, theta_init, weights, sigmas, optimizer, optimizer_kwargs = {}):
    
    sim_fun = lambda theta: simulate_all(theta, hamiltonian_operators, exp_settings)
    grad_fun = lambda theta: error_gradient(theta,
                                            theta_init,
                                            hamiltonian_operators,
                                            data_exp,
                                            sim_fun(theta),
                                            exp_settings,
                                            weights,
                                            sigmas)
    
    return find_parameters(data_exp, theta_init, weights, sigmas, sim_fun, grad_fun, optimizer, optimizer_kwargs)
    

### EXACT SIMULATION
##### Broken down into small functions so that caching can easily be added

@np_cache
def hamiltonian_model(theta, operators):
    return np.array(np.sum(operators*theta.reshape((len(theta),1,1)),axis = 0))

@np_cache
def unitary_evolution(hamiltonian, time):
    return expm(-1j*time*hamiltonian)

@np_cache
def apply_unitary(state, unitary):
    return np.linalg.multi_dot([unitary.conj().T, state, unitary])

@np_cache
def evolve(state, hamiltonian, time):
    evolution = unitary_evolution(hamiltonian, time)
    return apply_unitary(state, evolution)

@np_cache
def expected_value(observable, state):
    val = np.trace(np.dot(observable,state))
    val = val.item()
    return val

@np_cache
def simulate_one(theta, operators, state, time, observable):
    #TO DO: add noise at the end
    hamiltonian = hamiltonian_model(theta, operators)
    evolved_state = evolve(state, hamiltonian, time)
    return expected_value(observable, evolved_state).real

def simulate_all(theta, hamiltonian_operators, exp_settings):
    data = np.array([simulate_one(theta, hamiltonian_operators,
                                  np.array(s['state']),
                                  np.array(s['time']),
                                  np.array(s['observable']))
                                      for s in exp_settings])
    return data


def clear_all_caches():
    for fun in [simulate_one,
                expected_value,
                evolve,
                apply_unitary,
                unitary_evolution,
                hamiltonian_model]:
        fun.cache_clear()

### GRADIENT CALCULATION
##### TO DO: For now this works with exact simulation; should be generalized so that circuits etc can be used

@np_cache
def commutator(x,y):
    return np.dot(x,y) - np.dot(y,x)

def j(observable, evolved_state, hamiltonian, operator, time):
    evolved_operator = evolve(operator, hamiltonian, time)
    com = commutator(evolved_operator, evolved_state)
    return expected_value(observable, com)

def J(observable, evolved_state, hamiltonian, operator, time):
    fun = lambda s: j(observable, evolved_state, hamiltonian, operator, time - s)
    fun_re = lambda s: fun(s).real
    fun_im = lambda s: fun(s).imag
    re = integrate.quad(fun_re, 0, time)[0]
    im = integrate.quad(fun_im, 0, time)[0]
    return re+1j*im

def Js(hamiltonian, operators, exp_settings):
    J_array = []
    for s in exp_settings:
        observable = s['observable']
        time = s['time']
        state = s['state']
        evolved_state = evolve(state, hamiltonian, time)
        J_array.append([J(observable, evolved_state, hamiltonian, operator, time)
                        for operator in operators
                       ])
    J_array = np.array(J_array).T
    
    return J_array

def error_gradient(theta, theta0, hamiltonian_operators, data_exp, data_sim, exp_settings, weights, sigmas):
    
    term1 = (theta-theta0)/weights
    
    data_diff = (data_sim - data_exp)/sigmas
   
    J_array = Js(hamiltonian_model(theta, hamiltonian_operators), hamiltonian_operators, exp_settings)
    
    term2 = -np.sum(data_diff*J_array, axis = 1).imag
    
    return term1+term2
