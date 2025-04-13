import numpy as np
import matplotlib.pyplot as plt

from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import SparsePauliOp, Pauli 


class QuantumOptimizer:
    def __init__(self, instance, n, K):
        self.instance = instance
        self.n = n
        self.K = K

    def construct_operator(self,Q, g, c):
        """Construye un SparsePauliOp desde el QUBO (Q, g, c)"""
        num_vars = Q.shape[0]
        paulis = []
        coeffs = []

        for i in range(num_vars):
            z = ['I'] * num_vars
            z[i] = 'Z'
            paulis.append(Pauli(''.join(z)))
            coeffs.append(-0.5 * g[i])

        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                if Q[i, j] != 0:
                    z = ['I'] * num_vars
                    z[i] = 'Z'
                    z[j] = 'Z'
                    paulis.append(Pauli(''.join(z)))
                    coeffs.append(0.25 * Q[i, j])

        return SparsePauliOp(paulis, coeffs) + SparsePauliOp.from_list([("I" * num_vars, c)])


    def binary_representation(self, x_sol=0):
        instance = self.instance
        n = self.n
        K = self.K

        A = np.max(instance) * 100

        instance_vec = instance.reshape(n**2)
        w_list = [instance_vec[x] for x in range(n**2) if instance_vec[x] > 0]
        w = np.zeros(n * (n - 1))
        for ii in range(len(w_list)):
            w[ii] = w_list[ii]

        Id_n = np.eye(n)
        Im_n_1 = np.ones([n - 1, n - 1])
        Iv_n_1 = np.ones(n)
        Iv_n_1[0] = 0
        Iv_n = np.ones(n - 1)
        neg_Iv_n_1 = np.ones(n) - Iv_n_1

        v = np.zeros([n, n * (n - 1)])
        for ii in range(n):
            count = ii - 1
            for jj in range(n * (n - 1)):
                if jj // (n - 1) == ii:
                    count = ii
                if jj // (n - 1) != ii and jj % (n - 1) == count:
                    v[ii][jj] = 1.0

        vn = np.sum(v[1:], axis=0)
        Q = A * (np.kron(Id_n, Im_n_1) + np.dot(v.T, v))
        g = (
            w
            - 2 * A * (np.kron(Iv_n_1, Iv_n) + vn.T)
            - 2 * A * K * (np.kron(neg_Iv_n_1, Iv_n) + v[0].T)
        )
        c = 2 * A * (n - 1) + 2 * A * (K**2)

        try:
            max(x_sol)
            fun = (
                lambda x: np.dot(np.around(x), np.dot(Q, np.around(x)))
                + np.dot(g, np.around(x))
                + c
            )
            cost = fun(x_sol)
        except:
            cost = 0

        return Q, g, c, cost

    def construct_problem(self, Q, g, c) -> QuadraticProgram:
        qp = QuadraticProgram()
        for i in range(self.n * (self.n - 1)):
            qp.binary_var(str(i))
        qp.objective.quadratic = Q
        qp.objective.linear = g
        qp.objective.constant = c
        return qp

    def solve_problem(self, qp):
        algorithm_globals.random_seed = 10598

        Q, g, c, _ = self.binary_representation()
        operator = self.construct_operator(Q, g, c)

        vqe = SamplingVQE(sampler=Sampler(), optimizer=SPSA(), ansatz=RealAmplitudes())
        result = vqe.compute_minimum_eigenvalue(operator)

        x_sol = np.round(result.optimal_point).astype(int)

        _, _, _, level = self.binary_representation(x_sol=x_sol)
        return x_sol, level


# ========================
# Datos de entrada simulados
# ========================
n = 4
K = 2
instance = np.random.randint(1, 10, size=(n, n))
np.fill_diagonal(instance, 0)

# Para comparación clásica (ficticia)
x = None
z = None
classical_cost = 0

# Crear optimizador
quantum_optimizer = QuantumOptimizer(instance, n, K)

# Evaluar función binaria
try:
    if z is not None:
        Q, g, c, binary_cost = quantum_optimizer.binary_representation(x_sol=z)
        print("Binary cost:", binary_cost, "classical cost:", classical_cost)
        if np.abs(binary_cost - classical_cost) < 0.01:
            print("Binary formulation is correct")
        else:
            print("Error in the binary formulation")
    else:
        print("CPLEX no disponible, ejecutando sin comparación clásica.")
        Q, g, c, binary_cost = quantum_optimizer.binary_representation()
        print("Binary cost:", binary_cost)
except NameError as e:
    print("Error: variables no definidas")
    print(e)

# Construir y resolver el problema
qp = quantum_optimizer.construct_problem(Q, g, c)
quantum_solution, quantum_cost = quantum_optimizer.solve_problem(qp)

print("\nQuantum solution:", quantum_solution)
print("Quantum cost:", quantum_cost)

# Convertir solución para graficar
x_quantum = np.zeros(n**2)
kk = 0
for ii in range(n**2):
    if ii // n != ii % n:
        x_quantum[ii] = quantum_solution[kk]
        kk += 1


# ========================
# Visualización simple
# ========================
def visualize_solution(xc, yc, x_sol, cost, n, K, title="Quantum"):
    plt.figure(figsize=(6, 6))
    plt.title(f"{title} Solution - Cost: {cost:.2f}")
    plt.imshow(x_sol.reshape(n, n), cmap="Blues")
    plt.colorbar()
    plt.show()


# Generar puntos ficticios para la visualización
xc = np.random.rand(n)
yc = np.random.rand(n)

visualize_solution(xc, yc, x_quantum, quantum_cost, n, K, "Quantum")

if x is not None:
    visualize_solution(xc, yc, x, classical_cost, n, K, "Classical")
