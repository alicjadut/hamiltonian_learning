import numpy as np
from scipy.optimize import fmin_cg
from scipy.linalg import expm
from scipy import integrate




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

def find_parameters(data_exp, theta_init, weights, sigmas, sim_fun, grad_fun):
    return fmin_cg(
        lambda theta: parameter_error(theta, theta_init, sim_fun(theta), data_exp, weights, sigmas),
        theta_init,
        grad_fun
    )

def hamiltonian_learning(hamiltonian_operators, exp_settings, data_exp, theta_init, weights, sigmas):
    
    sim_fun = lambda theta: simulate_all(theta, hamiltonian_operators, exp_settings)
    grad_fun = lambda theta: error_gradient(theta,
                                            theta_init,
                                            hamiltonian_operators,
                                            data_exp,
                                            sim_fun(theta),
                                            exp_settings,
                                            weights,
                                            sigmas)
    
    return find_parameters(data_exp, theta_init, weights, sigmas, sim_fun, grad_fun)
    

### EXACT SIMULATION
##### Broken down into small functions so that caching can easily be added

def hamiltonian_model(theta, operators):
    return np.matrix(np.sum(np.array(operators)*theta.reshape((len(theta),1,1)),axis = 0))

def unitary_evolution(ham_matrix, time):
    return np.matrix(expm(-1j*time*ham_matrix))
    
def apply_unitary(state, unitary):
    return unitary.H * state * unitary

def evolve(state, ham_matrix, time):
    evolution = unitary_evolution(ham_matrix, time)
    return apply_unitary(state, evolution)

def expected_value(observable, state):
    val = np.trace(np.dot(observable,state))
    val = val.item()
    return val



def simulate_one(theta, operators, state, time, observable):
    #TO DO: add noise at the end
    hamiltonian = hamiltonian_model(theta, operators)
    evolved_state = evolve(state, hamiltonian, time)
    return expected_value(observable, evolved_state).real

def simulate_all(theta, hamiltonian_operators, exp_settings):
    data = np.array([simulate_one(theta, hamiltonian_operators, s['state'], s['time'], s['observable'])
                                      for s in exp_settings])
    return data


### GRADIENT CALCULATION
##### TO DO: For now this works with exact simulation; should be generalized so that circuits etc can be used

def commutator(x,y):
    return x*y - y*x

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