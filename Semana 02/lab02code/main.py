import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GeneticAlgorithm:
    def __init__(
        self,
        pop_size=30,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        patience=10,
        min_delta=1e-3,
        seed=None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.patience = patience
        self.min_delta = min_delta
        self.best_fitness_per_generation = []
        self.global_best = None
        self.global_best_fit = -float("inf")

    def run(self, ml_task):
        population = [ml_task.create_individual() for _ in range(self.pop_size)]
        global_best = None
        global_best_fit = -float('inf')
        global_best_sse = float('inf')
        best_sse_so_far = float('inf')
        no_improvement_counter = 0

        for gen in range(self.generations):
            fitnesses = [ml_task.fitness_function(ind) for ind in population]
            best_idx = np.argmax(fitnesses)
            best_ind = population[best_idx]
            best_fit = fitnesses[best_idx]
            current_sse = ml_task.calculate_sse(best_ind)

            if current_sse < global_best_sse:
                global_best = best_ind
                global_best_fit = best_fit
                global_best_sse = current_sse

            if current_sse < best_sse_so_far - self.min_delta:
                best_sse_so_far = current_sse
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= self.patience:
                print(f"Early stopping at generation {gen} due to no SSE improvement.")
                break

            self.best_fitness_per_generation.append(-best_fit)

            selected = self.tournament_selection(population, fitnesses)

            new_pop = []
            for i in range(0, len(selected), 2):
                p1 = selected[i]
                p2 = selected[(i+1) % len(selected)]
                c1, c2 = ml_task.crossover(p1, p2, self.crossover_rate)
                c1 = ml_task.mutation(c1, gen, self.generations)
                c2 = ml_task.mutation(c2, gen, self.generations)
                new_pop.append(c1)
                new_pop.append(c2)

            population = new_pop[:self.pop_size]

        self.plot_clusters(global_best, ml_task)

        return global_best

    def tournament_selection(self, population, fitnesses, k=3):
        selected = []
        zipped = list(zip(population, fitnesses))
        for _ in range(len(population)):
            tournament = random.sample(zipped, k)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def plot_clusters(self, best_solution, ml_task, dim_x=0, dim_y=1):
        data = ml_task.data
        centers = np.array(best_solution).reshape(ml_task.k, ml_task.dim)

        dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        cluster_assignments = np.argmin(dists, axis=1)

        plt.figure(figsize=(8, 6))
        for i in range(ml_task.k):
            plt.scatter(data[cluster_assignments == i, dim_x], data[cluster_assignments == i, dim_y], label=f'Cluster {i}')

        plt.scatter(centers[:, dim_x], centers[:, dim_y], c='red', marker='x', s=200, label='Centroides')
        plt.xlabel(f'Dimension {dim_x}')
        plt.ylabel(f'Dimension {dim_y}')
        plt.title('Clustering con Algoritmo Genético')
        plt.legend()
        plt.show()


class MachineLearningTask:
    def __init__(self, data, k=3, mutation_rate=0.05):
        self.data = np.array(data)
        self.k = k
        self.dim = data.shape[1]
        self.params_per_ind = self.k * self.dim
        self.lower_bound = np.min(self.data) - 10
        self.upper_bound = np.max(self.data) + 10
        self.mutation_rate = mutation_rate

    def create_individual(self):
        return tuple(
            random.uniform(self.lower_bound, self.upper_bound)
            for _ in range(self.params_per_ind)
        )

    def calculate_sse(self, individual):
        centers = np.array(individual).reshape(self.k, self.dim)
        dists = np.linalg.norm(self.data[:, None, :] - centers[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        return np.sum(min_dists**2)

    def fitness_function(self, individual):
        return -self.calculate_sse(individual)

    def crossover(self, parent1, parent2, crossover_rate):
        if random.random() > crossover_rate:
            return parent1, parent2
        alpha = random.random()
        child1 = tuple(alpha * x1 + (1 - alpha) * x2 for x1, x2 in zip(parent1, parent2))
        child2 = tuple(alpha * x2 + (1 - alpha) * x1 for x1, x2 in zip(parent1, parent2))
        return child1, child2

    def mutation(self, individual, generation, max_gens):
        adaptive_rate = self.mutation_rate * (1.0 - float(generation) / max_gens)
        ind_list = list(individual)
        for i in range(len(ind_list)):
            if random.random() < adaptive_rate:
                shift = random.uniform(-1, 1)
                ind_list[i] += shift
                ind_list[i] = max(min(ind_list[i], self.upper_bound), self.lower_bound)
        return tuple(ind_list)


def plot_fitness_evolution(best_fitness_per_generation):
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_per_generation, marker='o')
    plt.xlabel('Generaciones')
    plt.ylabel('Fitness (-SSE)')
    plt.title('Evolución del Fitness por Generación')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    file_path = "./Mall_Customers.csv"
    df = pd.read_csv(file_path)

    df_clustering = df.iloc[:, 1:]  # Excluir columna ID
    df_clustering['Gender'] = df_clustering['Gender'].map({'Female': 0, 'Male': 1})
    df_clustering.columns = ['Gender', 'Age', 'Annual_Income', 'Spending_Score']

    ga = GeneticAlgorithm(
        pop_size=30,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.2,
        patience=7,
        min_delta=0.01
    )

    ml_task = MachineLearningTask(data=df_clustering[['Annual_Income', 'Spending_Score']].values, k=3)

    best_solution = ga.run(ml_task)

    print("\n=== Mejor Solución ===")
    centers = np.array(best_solution).reshape(ml_task.k, ml_task.dim)
    print("Centroides:")
    print(centers)

    plot_fitness_evolution(ga.best_fitness_per_generation)
