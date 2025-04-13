import numpy as np
import matplotlib.pyplot as plt
import math
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpBinary, LpContinuous, PULP_CBC_CMD


# Qiskit moderno (por si lo querés mezclar con cuántico después)
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler


# Inicialización del problema
n = 4  # número de nodos
K = 2  # número de vehículos


# ============================
# Generar instancia aleatoria
# ============================
class Initializer:
    def __init__(self, n):
        self.n = n

    def generate_instance(self):
        np.random.seed(1543)
        xc = (np.random.rand(self.n) - 0.5) * 10
        yc = (np.random.rand(self.n) - 0.5) * 10

        instance = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist = (xc[i] - xc[j]) ** 2 + (yc[i] - yc[j]) ** 2
                instance[i, j] = dist
                instance[j, i] = dist
        return xc, yc, instance


initializer = Initializer(n)
xc, yc, instance = initializer.generate_instance()


# ============================
# Solver clásico con PuLP
# ============================
class ClassicalOptimizerPuLP:
    def __init__(self, instance, n, K):
        self.instance = instance
        self.n = n
        self.K = K

    def compute_allowed_combinations(self):
        f = math.factorial
        return f(self.n) / f(self.K) / f(self.n - self.K)

    def solve_with_pulp(self):
        n = self.n
        K = self.K
        instance = self.instance

        # Variables
        x = {(i, j): LpVariable(f"x_{i}_{j}", cat=LpBinary) for i in range(n) for j in range(n) if i != j}
        u = {i: LpVariable(f"u_{i}", lowBound=0.1, cat=LpContinuous) for i in range(1, n)}

        # Problema
        prob = LpProblem("CVRP", LpMinimize)

        # Función objetivo
        prob += lpSum(instance[i, j] * x[i, j] for i in range(n) for j in range(n) if i != j)

        # Restricciones de entrada y salida
        for i in range(1, n):
            prob += lpSum(x[i, j] for j in range(n) if j != i) == 1
            prob += lpSum(x[j, i] for j in range(n) if j != i) == 1

        # Salidas y entradas del depósito
        prob += lpSum(x[0, j] for j in range(1, n)) == K
        prob += lpSum(x[j, 0] for j in range(1, n)) == K

        # Sub-tour elimination (MTZ)
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    prob += u[i] - u[j] + (n - 1) * x[i, j] <= n - 2

        # Resolver
        prob.solve(PULP_CBC_CMD(msg=False))

        # Extraer solución
        solution = np.zeros(n * n)
        for i in range(n):
            for j in range(n):
                if i != j and x[i, j].value() == 1:
                    solution[i * n + j] = 1

        cost = sum(instance[i, j] * x[i, j].value() for i in range(n) for j in range(n) if i != j)
        return solution, cost


# ============================
# Visualización
# ============================
def visualize_solution(xc, yc, x, cost, n, K, title_str):
    plt.figure()
    plt.scatter(xc, yc, s=200)
    for i in range(len(xc)):
        plt.annotate(i, (xc[i] + 0.15, yc[i]), size=16, color="r")
    plt.plot(xc[0], yc[0], "r*", ms=20)
    plt.grid()

    for ii in range(n * n):
        if x[ii] > 0:
            ix = ii // n
            iy = ii % n
            plt.arrow(
                xc[ix],
                yc[ix],
                xc[iy] - xc[ix],
                yc[iy] - yc[ix],
                length_includes_head=True,
                head_width=0.25,
            )

    plt.title(title_str + f" cost = {cost:.2f}")
    plt.show()


# ============================
# Ejecutar optimizador
# ============================
optimizer = ClassicalOptimizerPuLP(instance, n, K)
print("Number of feasible solutions =", optimizer.compute_allowed_combinations())

x, classical_cost = optimizer.solve_with_pulp()
print("Solution (flattened):", x)

if x is not None:
    visualize_solution(xc, yc, x, classical_cost, n, K, "Classical (PuLP)")
