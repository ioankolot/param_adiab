from qiskit.visualization import *
import numpy as np
import matplotlib.pyplot as plt
from quantum_circuit import QCir
from qiskit.quantum_info import Statevector
import scipy.optimize as optimize
import networkx as nx
import collections
from qiskit.aqua.operators.gradients import Hessian
from qiskit.quantum_info.operators import Operator

#We start by defining the Pauli Matrices.
pauliz = np.array([[1, 0], [0, -1]])
paulix = np.array([[0, 1], [1, 0]])
pauliy = np.array([[0, -1j], [1j, 0]])
identity = np.array([[1, 0], [0, 1]])

number_of_qubits = 10
number_of_parameters = 3*number_of_qubits #This should change depending on the ansatz family
ansatz = 'rycxrzcxrz' #Choose the ansatz family (you can choose different architectures, check quantum_circuit.py )
steps = 50 #Choose number of steps to interpolate from initial to final Hamiltonian
connectivity = 'nearest-neighbors' #This is the connectivity of the non-parameterized gates in the Hardware Efficient ansatz

graph = nx.random_regular_graph(3, number_of_qubits, seed=20)
w = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))


def tensor_pauli(number_of_qubits, which_qubit, pauli_matrix): #This matrix represents a Pauli matrix acting in a single qubit in high dimensional Hilbert Spaces

    if which_qubit == 0:
        matrix = pauli_matrix
    else:
        matrix = identity

    for qubit in range(1, number_of_qubits):
        if which_qubit == qubit:
            matrix = np.kron(pauli_matrix, matrix)
        else:
            matrix = np.kron(identity, matrix)

    return matrix



def initial_hamiltonian(number_of_qubits): #Here we define the initial Hamiltonian which we choose it to be -sigma_x for all qubits.

    initial_ham = np.zeros((2**number_of_qubits, 2**number_of_qubits))
    for qubit in range(number_of_qubits):
        initial_ham -= tensor_pauli(number_of_qubits, qubit, paulix)

    return initial_ham



def maxcut_hamiltonian(adjacency_matrix): #This functions creates the matrix representation (2^n x 2^n) of the MaxCut Hamiltonian.

    Hamiltonian = np.zeros((2**number_of_qubits, 2**number_of_qubits))
    for vertex1 in range(number_of_qubits):
        for vertex2 in range(number_of_qubits):
            if vertex1 < vertex2:
                if w[vertex1, vertex2] != 0:
                    Hamiltonian += 1/2*np.dot(tensor_pauli(number_of_qubits, vertex1, pauliz), tensor_pauli(number_of_qubits, vertex2, pauliz))

    return Hamiltonian



H_0 = initial_hamiltonian(number_of_qubits)
H_1 = maxcut_hamiltonian(w)

def choose_initial_optimal_thetas(number_of_qubits, ansatz_family): #This function returns the initial optimal angles depending on the ansatz.

    init_thetas = []

    if ansatz_family == 'rycxrz':
        for qubit in range(number_of_qubits):
            init_thetas.append(np.pi/2)

        for qubit in range(number_of_qubits):
            init_thetas.append(0)

    elif ansatz_family == 'ryczry':
        for qubit in range(number_of_qubits):
            init_thetas.append(0)

        for qubit in range(number_of_qubits):
            init_thetas.append(np.pi/2)

    elif ansatz_family == 'rycxrzcxrz':
        for qubit in range(number_of_qubits):
            init_thetas.append(np.pi/2)

        for qubit in range(2*number_of_qubits):
            init_thetas.append(0)



    return init_thetas


def best_cost_brute(adjacency_matrix): #This function calculates the optimal cost function by brute force
    best_cost = 0
    number_of_qubits = len(adjacency_matrix)
    best_string = 0
    costs = collections.defaultdict(list)
    for b in range(2**number_of_qubits):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(number_of_qubits)))]
        cost = 0
        for i in range(number_of_qubits):
            for j in range(number_of_qubits):
                cost += adjacency_matrix[i,j] * x[i] * (1-x[j])
        
        x.reverse()
        costs[cost].append(x)

        if best_cost < cost:
            best_cost = cost
            best_string = x

    costs = sorted(costs.items())
    return best_cost, best_string, costs

best_cost, best_string, costs = best_cost_brute(w)
print(costs)
print(f'For the given instance the optimal cost is {best_cost} and the bitstrings corresponding to that are {costs[-1][1]}')


def calculate_expectation_value(number_of_qubits, matrix, thetas, connectivity, ansatz_family): #This function calculates the expectation value of a given observable
    circuit = QCir(number_of_qubits, thetas, connectivity, ansatz_family)
    sv1 = Statevector.from_label('0'*number_of_qubits)
    sv1 = sv1.evolve(circuit.qcir)
    expectation_value = sv1.expectation_value(matrix)
    return expectation_value


def calculate_instantaneous_hamiltonian(time):
    return (1-time)*H_0 + time*H_1



def get_offset(graph): #THis is the constant part (for unweighted graphs) in the MaxCut Hamiltonian
    return graph.number_of_edges()/2



thetas = choose_initial_optimal_thetas(number_of_qubits, ansatz)
offset = get_offset(graph)


#We must first calculate the linear system of equations.

def calculate_linear_system(number_of_qubits, number_of_parameters, hamiltonian, thetas, connectivity='nearest-neighbors', ansatz_family=ansatz): #This function calculates the linear system of equations

    zero_order_terms = np.zeros((number_of_parameters,))
    first_order_terms = np.zeros((number_of_parameters, number_of_parameters))


    for parameter in range(number_of_parameters):

        zero_order_thetas_1, zero_order_thetas_2 = thetas.copy(), thetas.copy()
        zero_order_thetas_1[parameter] += np.pi/2
        zero_order_thetas_2[parameter] -= np.pi/2


        zero_order_terms[parameter] += 1/2*calculate_expectation_value(number_of_qubits, hamiltonian, zero_order_thetas_1, connectivity, ansatz_family)
        zero_order_terms[parameter] -= 1/2*calculate_expectation_value(number_of_qubits, hamiltonian, zero_order_thetas_2, connectivity, ansatz_family)


    #Next we continue with second order terms.
    for parameter1 in range(number_of_parameters):
        for parameter2 in range(number_of_parameters):
            if parameter1 <= parameter2:
                
                first_order_thetas_1, first_order_thetas_2, first_order_thetas_3, first_order_thetas_4 = thetas.copy(), thetas.copy(), thetas.copy(), thetas.copy()

                first_order_thetas_1[parameter1] += np.pi/2
                first_order_thetas_1[parameter2] += np.pi/2


                first_order_thetas_2[parameter1] += np.pi/2
                first_order_thetas_2[parameter2] -= np.pi/2

                first_order_thetas_3[parameter1] -= np.pi/2
                first_order_thetas_3[parameter2] += np.pi/2

                first_order_thetas_4[parameter1] -= np.pi/2
                first_order_thetas_4[parameter2] -= np.pi/2

                first_order_terms[parameter1, parameter2] += calculate_expectation_value(number_of_qubits, hamiltonian, first_order_thetas_1, connectivity, ansatz_family)/4
                first_order_terms[parameter1, parameter2] -= calculate_expectation_value(number_of_qubits, hamiltonian, first_order_thetas_2, connectivity, ansatz_family)/4
                first_order_terms[parameter1, parameter2] -= calculate_expectation_value(number_of_qubits, hamiltonian, first_order_thetas_3, connectivity, ansatz_family)/4
                first_order_terms[parameter1, parameter2] += calculate_expectation_value(number_of_qubits, hamiltonian, first_order_thetas_4, connectivity, ansatz_family)/4

                first_order_terms[parameter2, parameter1] = first_order_terms[parameter1, parameter2]

    return zero_order_terms, first_order_terms


def calculate_hessian_matrix(number_of_qubits, number_of_parameters, hamiltonian, thetas, connectivity='nearest-neighbors', ansatz_family=ansatz):

    hessian = np.zeros((number_of_parameters, number_of_parameters))
    
    for parameter1 in range(number_of_parameters):
        for parameter2 in range(number_of_parameters):
            if parameter1 < parameter2:
                
                hessian_thetas_1, hessian_thetas_2, hessian_thetas_3, hessian_thetas_4 = thetas.copy(), thetas.copy(), thetas.copy(), thetas.copy()

                hessian_thetas_1[parameter1] += np.pi/2
                hessian_thetas_1[parameter2] += np.pi/2

                hessian_thetas_2[parameter1] -= np.pi/2
                hessian_thetas_2[parameter2] += np.pi/2

                hessian_thetas_3[parameter1] += np.pi/2
                hessian_thetas_3[parameter2] -= np.pi/2

                hessian_thetas_4[parameter1] -= np.pi/2
                hessian_thetas_4[parameter2] -= np.pi/2

                hessian[parameter1, parameter2] += calculate_expectation_value(number_of_qubits, hamiltonian, hessian_thetas_1, connectivity, ansatz_family)/4
                hessian[parameter1, parameter2] -= calculate_expectation_value(number_of_qubits, hamiltonian, hessian_thetas_2, connectivity, ansatz_family)/4
                hessian[parameter1, parameter2] -= calculate_expectation_value(number_of_qubits, hamiltonian, hessian_thetas_3, connectivity, ansatz_family)/4
                hessian[parameter1, parameter2] += calculate_expectation_value(number_of_qubits, hamiltonian, hessian_thetas_4, connectivity, ansatz_family)/4

                hessian[parameter2, parameter1] = hessian[parameter1, parameter2]
                
            if parameter1 == parameter2:

                hessian_thetas_1 , hessian_thetas_2 = thetas.copy(), thetas.copy()

                hessian_thetas_1[parameter1] += np.pi
                hessian_thetas_2[parameter1] -= np.pi
                
                hessian[parameter1, parameter1] += calculate_expectation_value(number_of_qubits, hamiltonian, hessian_thetas_1, connectivity, ansatz_family)/4
                hessian[parameter1, parameter1] += calculate_expectation_value(number_of_qubits, hamiltonian, hessian_thetas_2, connectivity, ansatz_family)/4
                hessian[parameter1, parameter1] -= calculate_expectation_value(number_of_qubits, hamiltonian, thetas, connectivity, ansatz_family)/2

    w, v = np.linalg.eig(hessian)    
    diagonal_hessian = np.zeros((number_of_parameters, number_of_parameters))
    np.fill_diagonal(diagonal_hessian, w)

    return diagonal_hessian


def minimum_eigenvalue(matrix): #Find the minimum eigenvalue of a diagonal matrix. This is used as the constraint to be positive semidefinite 

    point = 0
    minim_eig = np.infty
    for i in range(len(matrix)):
        if matrix[i,i] < minim_eig:
            minim_eig = matrix[i,i]
            point = i

    print(minim_eig)

    return round(matrix[point, point], 5)


'''
#WE DONT USE THAT YET. 
def calculate_third_order_derivatives(number_of_qubits, number_of_parameters, hamiltonian, thetas, connectivity='nearest-neighbors', adjacency_matrix=w , ansatz_family=ansatz):
    third_derivatives = np.zeros((number_of_parameters, number_of_parameters, number_of_parameters))

    for parameter1 in range(number_of_parameters):
        for parameter2 in range(number_of_parameters):
            for parameter3 in range(number_of_parameters):

                third_order_thetas1, third_order_thetas2, third_order_thetas3, third_order_thetas4, third_order_thetas5, third_order_thetas6, third_order_thetas7, third_order_thetas8 = thetas.copy(), thetas.copy(), thetas.copy(), thetas.copy(), thetas.copy(), thetas.copy(), thetas.copy(), thetas.copy()

                third_order_thetas1[parameter1] += np.pi/2
                third_order_thetas1[parameter2] += np.pi/2
                third_order_thetas1[parameter3] += np.pi/2

                third_order_thetas2[parameter1] += np.pi/2
                third_order_thetas2[parameter2] += np.pi/2
                third_order_thetas2[parameter3] -= np.pi/2

                third_order_thetas3[parameter1] -= np.pi/2
                third_order_thetas3[parameter2] += np.pi/2
                third_order_thetas3[parameter3] += np.pi/2

                third_order_thetas4[parameter1] -= np.pi/2
                third_order_thetas4[parameter2] += np.pi/2
                third_order_thetas4[parameter3] -= np.pi/2

                third_order_thetas5[parameter1] += np.pi/2
                third_order_thetas5[parameter2] -= np.pi/2
                third_order_thetas5[parameter3] += np.pi/2

                third_order_thetas6[parameter1] += np.pi/2
                third_order_thetas6[parameter2] -= np.pi/2
                third_order_thetas6[parameter3] -= np.pi/2

                third_order_thetas7[parameter1] -= np.pi/2
                third_order_thetas7[parameter2] -= np.pi/2
                third_order_thetas7[parameter3] += np.pi/2

                third_order_thetas8[parameter1] -= np.pi/2
                third_order_thetas8[parameter2] -= np.pi/2
                third_order_thetas8[parameter3] -= np.pi/2

                third_derivatives[parameter1, parameter2, parameter3] += calculate_expectation_value(number_of_qubits, hamiltonian, third_order_thetas1, connectivity, adjacency_matrix, ansatz_family)/8
                third_derivatives[parameter1, parameter2, parameter3] -= calculate_expectation_value(number_of_qubits, hamiltonian, third_order_thetas2, connectivity, adjacency_matrix, ansatz_family)/8  
                third_derivatives[parameter1, parameter2, parameter3] -= calculate_expectation_value(number_of_qubits, hamiltonian, third_order_thetas3, connectivity, adjacency_matrix, ansatz_family)/8
                third_derivatives[parameter1, parameter2, parameter3] += calculate_expectation_value(number_of_qubits, hamiltonian, third_order_thetas4, connectivity, adjacency_matrix, ansatz_family)/8
                third_derivatives[parameter1, parameter2, parameter3] -= calculate_expectation_value(number_of_qubits, hamiltonian, third_order_thetas5, connectivity, adjacency_matrix, ansatz_family)/8
                third_derivatives[parameter1, parameter2, parameter3] += calculate_expectation_value(number_of_qubits, hamiltonian, third_order_thetas6, connectivity, adjacency_matrix, ansatz_family)/8
                third_derivatives[parameter1, parameter2, parameter3] += calculate_expectation_value(number_of_qubits, hamiltonian, third_order_thetas7, connectivity, adjacency_matrix, ansatz_family)/8
                third_derivatives[parameter1, parameter2, parameter3] -= calculate_expectation_value(number_of_qubits, hamiltonian, third_order_thetas8, connectivity, adjacency_matrix, ansatz_family)/8


    return third_derivatives

def calculate_hessian_matrix_alternative(matrix1, matrix2, epsilons):

    hessian_matrix = np.zeros((number_of_parameters, number_of_parameters))

    for parameter1 in range(number_of_parameters):
        for parameter2 in range(number_of_parameters):

            hessian_matrix[parameter1, parameter2] += matrix1[parameter1, parameter2]

            for epsilon in range(number_of_parameters):
                hessian_matrix[parameter1, parameter2] += epsilons[epsilon]*matrix2[parameter1, parameter2, epsilon]


    w, v = np.linalg.eig(hessian_matrix)    
    diagonal_hessian = np.zeros((number_of_parameters, number_of_parameters))
    np.fill_diagonal(diagonal_hessian, w)


    return diagonal_hessian
'''


def adiabatic_solver(steps, angles, ansatz_family = ansatz): #This is the main function that step by step interpolates from the initial Hamiltonian H_0 to the final Hamiltonian H_1

    lambdas = [np.round(i, 5) for i in np.linspace(0, 1, steps)]
    optimal_thetas = angles.copy()
    print(f'We start with the optimal angles of the initial hamiltonian: {optimal_thetas}')

    for lamda in lambdas[1:]:
        print('\n')
        print(f'We are working on {lamda}')
        hamiltonian = calculate_instantaneous_hamiltonian(lamda)

        zero, first = calculate_linear_system(number_of_qubits, number_of_parameters, hamiltonian, optimal_thetas)
        print(zero, first)



        def equations(x, terms={'0':zero, '1':first}): 
            zero = terms['0']
            first = terms['1']

            array = np.array([x[_] for _ in range(number_of_parameters)])
            equations = zero + np.dot(first, array)

            y = np.array([equations[_] for _ in range(number_of_parameters)])
            return -np.dot(y, y)


        #def minim_eig_constraint(x):
        #    new_thetas = [optimal_thetas[i] + x[i] for i in range(number_of_parameters)]
        #    return minimum_eigenvalue(calculate_hessian_matrix(number_of_qubits, number_of_parameters, hamiltonian, new_thetas))
        
        def small_values(x):
            return np.dot(x,x)

        cons = [{'type': 'ineq', 'fun':equations}]#, {'type': 'ineq', 'fun':minim_eig_constraint}]


        res = optimize.minimize(small_values, x0 = [0 for _ in range(number_of_parameters)], method='COBYLA', constraints=cons, options={'disp': False, 'maxiter':350})
        epsilons = [res.x[_] for _ in range(number_of_parameters)]

        print(f'The solutions of equations are {epsilons}')
        optimal_thetas = [optimal_thetas[_] + epsilons[_] for _ in range(number_of_parameters)]
        print(f'and the instantaneous expectation values is {calculate_expectation_value(number_of_qubits, hamiltonian, optimal_thetas, "nearest-neighbors", ansatz_family)- lamda*offset}')


    circuit = QCir(number_of_qubits, optimal_thetas, "nearest-neighbors", ansatz_family)
    sv1 = Statevector.from_label('0'*number_of_qubits)
    sv1 = sv1.evolve(circuit.qcir)
    probabilities_dictionary = sv1.probabilities_dict()
    print(probabilities_dictionary)

    print(f'\n The final expectation value is {calculate_expectation_value(number_of_qubits, hamiltonian, optimal_thetas, "nearest-neighbors", ansatz_family) - offset}')

    optimal_probability = 0

    #We can print the total probability of sampling the optimal solution.
    best_cost_str = int(best_cost)
    for item in probabilities_dictionary:
        item = [int(t) for t in list(item)]
        #print(item)
        if item in costs[-1][1]:
            str1 = ''.join(str(e) for e in item)
            optimal_probability += probabilities_dictionary[str1]
    print(f'The probability of sampling the optimal solution is {optimal_probability}')

    return optimal_thetas, optimal_probability



adiabatic_solver(steps, thetas)