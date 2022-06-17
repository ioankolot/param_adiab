from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.visualization import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from   matplotlib import cm
from scipy.optimize import minimize


class QCir():
    def __init__(self, number_of_qubits, thetas, connectivity, ansatz_family):

        self.number_of_qubits = number_of_qubits
        self.thetas = thetas    
        self.shots = 10000        

        self.qreg = QuantumRegister(self.number_of_qubits, name = 'q')
        self.qcir = QuantumCircuit(self.number_of_qubits)


        if ansatz_family == "rycxrz":
            for qubit in range(self.number_of_qubits):
                self.qcir.ry(self.thetas[qubit], qubit)

            self.vqe.barrier()

            #We allow three different cases for the connectivity. nearest-neighbors, all-to-all, problem-specific
            if connectivity == 'nearest-neighbors':
                for qubit1 in range(self.number_of_qubits-1):
                    self.qcir.cnot(qubit1, qubit1+1)
                    
                self.qcir.cnot(self.number_of_qubits-1, 0)
            

            elif connectivity == 'all-to-all':
                for qubit1 in range(self.number_of_qubits):
                    for qubit2 in range(self.number_of_qubits):
                        if qubit1<qubit2:
                            self.qcir.cnot(qubit1, qubit2)


            self.qcir.barrier()

            for qubit in range(self.number_of_qubits):
                self.qcir.rz(self.thetas[qubit + self.number_of_qubits], qubit)

        elif ansatz_family == "ryczry":
            for qubit in range(self.number_of_qubits):
                self.qcir.ry(self.thetas[qubit], qubit)

            self.qcir.barrier()

            if connectivity == 'nearest-neighbors':
                for qubit1 in range(self.number_of_qubits-1):
                    self.qcir.cz(qubit1, qubit1+1)
                    
                self.qcir.cz(self.number_of_qubits-1, 0)
            

            elif connectivity == 'all-to-all':
                for qubit1 in range(self.number_of_qubits):
                    for qubit2 in range(self.number_of_qubits):
                        if qubit1<qubit2:
                            self.qcir.cz(qubit1, qubit2)

            self.qcir.barrier()

            for qubit in range(self.number_of_qubits):
                self.qcir.ry(self.thetas[qubit + self.number_of_qubits], qubit)


        elif ansatz_family == 'rycxrzcxrz':
            for qubit in range(self.number_of_qubits):
                self.qcir.ry(self.thetas[qubit], qubit)

            self.qcir.barrier()


            for qubit1 in range(self.number_of_qubits-1):
                self.qcir.cx(qubit1, qubit1+1)
                        
            self.qcir.cx(self.number_of_qubits-1, 0)
                

            self.qcir.barrier()

            for qubit in range(self.number_of_qubits):
                self.qcir.rz(self.thetas[qubit + self.number_of_qubits], qubit)

            for qubit1 in range(self.number_of_qubits-1):
                self.qcir.cx(qubit1, qubit1+1)
                        
            self.qcir.cx(self.number_of_qubits-1, 0)

            for qubit in range(self.number_of_qubits):
                self.qcir.rz(self.thetas[qubit + 2*self.number_of_qubits], qubit)



