import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import sklearn

class GeneticAlgorithm:
    """
    Un marco genérico de Algoritmo Genético.
    """

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
        """
        :param pop_size: tamaño de la población
        :param generations: número máximo de generaciones
        :param crossover_rate: probabilidad de realizar crossover
        :param mutation_rate: probabilidad de mutar cada gen
        :param patience: detención temprana si no hay mejora durante 'patience' generaciones
        :param min_delta: mínima mejora para resetear la paciencia
        :param seed: semilla aleatoria opcional
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.patience = patience
        self.min_delta = min_delta

        # Campos opcionales para seguimiento
        self.best_fitness_per_generation = []
        self.global_best = None
        self.global_best_fit = -float("inf")

    def run(self, ml_task):
        """
        Bucle principal del GA:
          - crear población inicial (mediante ml_task.create_individual)
          - evaluar fitness (ml_task.fitness_function)
          - selección, crossover, mutación
          - seguimiento del mejor global
          - detención temprana
        :param ml_task: instancia que proporciona create_individual(), fitness_function(),
                        crossover(), mutation().

        :return: mejor individuo encontrado y lista de valores de fitness por generación.
        """
        # TODO: Implementar los pasos del GA
        # 1) Crear población inicial
        # 2) Evaluar fitness
        # 3) Controlar mejores soluciones
        # 4) for gen in range(self.generations):
        #    - construir nueva población con selección -> crossover -> mutación
        #    - checks de detención temprana
        #    - registrar mejor fitness
        population = [ml_task.create_individual() for _ in range(self.pop_size)]

        # We'll track the best SSE (which we want to minimize) -> or track best fitness = -SSE (maximize).
        global_best = None
        global_best_fit = -float('inf')
        global_best_sse = float('inf')

        best_sse_so_far = float('inf')
        no_improvement_counter = 0

        # for table
        try:
            from prettytable import PrettyTable
            table = PrettyTable()
            table.field_names = ["Gen", "Representation", "Best Fitness", "Best SSE", "Centers (truncated)"]
        except ImportError:
            table = None

        for gen in range(self.generations):
            # Evaluate fitness
            fitnesses = [ml_task.fitness_function(ind) for ind in population]
            
            # Identify best of this generation
            best_idx = np.argmax(fitnesses)
            best_ind = population[best_idx]
            best_fit = fitnesses[best_idx]
            current_sse = ml_task.calculate_sse(best_ind)

            # Update global best
            if current_sse < global_best_sse:
                global_best = best_ind
                global_best_fit = best_fit
                global_best_sse = current_sse

            # Early stopping check
            if current_sse < best_sse_so_far - self.min_delta:
                best_sse_so_far = current_sse
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= self.patience:
                print(f"Early stopping at generation {gen} due to no SSE improvement.")
                break

            # Logging to table
            if table is not None:
                real_params = best_ind
                # Show only first few center coords to avoid giant columns
                truncated_str = ", ".join([f"{x:.2f}" for x in real_params[:6]]) + " ..."
                table.add_row([
                    gen,
                    f"{best_ind}",
                    f"{best_fit:.3f}",
                    f"{current_sse:.3f}",
                    truncated_str
                ])

            # Selection
            selected = self.tournament_selection(population, fitnesses)

            # Crossover & Mutation -> next gen
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

        if table is not None:
            print(table)

        # Final check among last population
        final_fitnesses = [ml_task.fitness_function(ind) for ind in population]
        final_best_idx = np.argmax(final_fitnesses)
        final_best_ind = population[final_best_idx]
        final_best_fit = final_fitnesses[final_best_idx]
        final_best_sse = ml_task.calculate_sse(final_best_ind)

        # Compare to global best across all gens
        if global_best is not None and global_best_sse < final_best_sse:
            truly_best = global_best
            truly_best_fit = global_best_fit
            truly_best_sse = global_best_sse
        else:
            truly_best = final_best_ind
            truly_best_fit = final_best_fit
            truly_best_sse = final_best_sse

        # Print final info
        print("\n=== Final Reported Best Clustering (Global) ===")
        real_params = truly_best
        print(f"Best Fitness (=-SSE): {truly_best_fit:.4f}")
        print(f"SSE: {truly_best_sse:.4f}")
        # Optionally print out all center coordinates:
        reshaped_centers = np.array(real_params).reshape(ml_task.k, ml_task.dim)
        print(f"Cluster Centers:\n{reshaped_centers}")

        # Plot final result in 2D
        self.plot_clusters(truly_best, ml_task)


        return truly_best

    def tournament_selection(self, population, fitnesses, k=3):
        selected = []
        zipped = list(zip(population, fitnesses))
        for _ in range(len(population)):
            tournament = random.sample(zipped, k)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected



    def plot_clusters(self, best_solution, ml_task):
        data = ml_task.data
        centers = np.array(best_solution).reshape(ml_task.k, ml_task.dim)

        dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        cluster_assignments = np.argmin(dists, axis=1)

        plt.figure(figsize=(8, 6))
        for i in range(ml_task.k):
            plt.scatter(data[cluster_assignments == i, 0], data[cluster_assignments == i, 1], label=f'Cluster {i}')

        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroides')
        plt.xlabel('Annual Income')
        plt.ylabel('Spending Score')
        plt.title('Clustering con Algoritmo Genético')
        plt.legend()
        plt.show()
### `machine_learning_task.py` (Esqueleto de Tarea de ML Especializada)


class MachineLearningTask:
    """
    Esta clase debe contener los datos y definir cómo:
      - crear un individuo
      - calcular el fitness
      - hacer crossover
      - mutar los individuos
    """

    def __init__(self, data="Mall_Customers.csv", k=3, mutation_rate=0.05, generations=50,):
        """
        :param data: podría ser un conjunto de datos de entrenamiento, o un array de features, etc.
        :param k: parámetro de ejemplo (número de clústeres u otro objetivo).
        """
        self.data = data
        self.k = k
        self.dim = data.shape[1]
        self.params_per_ind = self.k * self.dim
        self.lower_bound = -10
        self.upper_bound = 10

        self.mutation_rate = mutation_rate
        self.generations = generations

        self.data = np.array(data)
        # optionally: check self.data.shape[1] == dim, etc.


    def create_individual(self):
        """
        Retorna una solución (individuo) aleatoria.
        Ejemplos:
          - Para clustering: lista de k*dim floats aleatorios.
          - Para clasificación: conjunto de pesos o hiperparámetros.
          - Para regresión simbólica: estructura de árbol o ecuación linear.
        """
        return tuple(
                random.uniform(self.lower_bound, self.upper_bound)
                for _ in range(self.params_per_ind)
            )

    def calculate_sse(self, individual):
            """
            Utility to get the SSE for an individual's cluster centers.
            """
            centers_vals = individual
            centers = np.array(centers_vals).reshape(self.k, self.dim)
            dists = np.linalg.norm(self.data[:, None, :] - centers[None, :, :], axis=2)
            min_dists = np.min(dists, axis=1)
            sse = np.sum(min_dists**2)
            return sse
    def fitness_function(self, individual):
        """
        Evalúa la calidad del individuo y retorna
        un valor numérico (cuanto más alto, mejor).
        Ejemplos:
          - Clustering: -SSE (SSE negativo)
          - Clasificación: exactitud en validación
          - Regresión: -ECM
        """
        centers_vals = individual

        # Reshape centers_vals into (k, dim)
        centers = np.array(centers_vals).reshape(self.k, self.dim)

        # Sum of squared distances
        # for each point, find nearest center
        points = self.data
        # distances shape => (#points, #centers)
        dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        # find min distance for each point
        min_dists = np.min(dists, axis=1)
        sse = np.sum(min_dists**2)

        return -sse  # negate to maximize


    def crossover(self, parent1, parent2, crossover_rate):
        """
        Retorna dos 'hijos'. Tal vez no hacer nada si random.random() > crossover_rate.
        """
        alpha = random.random()
        child1 = tuple(alpha*x1 + (1-alpha)*x2 for x1,x2 in zip(parent1,parent2))
        child2 = tuple(alpha*x2 + (1-alpha)*x1 for x1,x2 in zip(parent1,parent2))
        return child1, child2


    def mutation(self, individual, generation, max_gens):
        adaptive_rate = self.mutation_rate * (1.0 - float(generation)/max_gens)
            
        ind_list = list(individual)
        for i in range(len(ind_list)):
            if random.random() < adaptive_rate:
                shift = random.uniform(-1, 1)  # simple shift
                ind_list[i] += shift
                # enforce bounds
                ind_list[i] = max(min(ind_list[i], self.upper_bound), self.lower_bound)
        return tuple(ind_list)


if __name__ == "__main__":
    print("GA Clustering code demo")
# Cargar el dataset
    file_path = "./Mall_Customers.csv"  
    df = pd.read_csv(file_path)

    # Eliminar la columna ID
    df_clustering = df.iloc[:, 1:]  # Ahora incluye Gender

    # Convertir Gender de texto a numérico usando un bucle
    for i in range(len(df_clustering)):
        if df_clustering.loc[i, "Gender"] == "Female":
            df_clustering.loc[i, "Gender"] = 0
        elif df_clustering.loc[i, "Gender"] == "Male":
            df_clustering.loc[i, "Gender"] = 1

    # Convertir la columna Gender a tipo numérico (int)
    df_clustering["Gender"] = df_clustering["Gender"].astype(int)

    # Renombrar las columnas
    df_clustering.columns = ["Gender", "Age", "Annual_Income", "Spending_Score"]

    # Verificar los datos
    print(df_clustering.head())



    # Example usage for 2D data, real-coded representation
    ga_cluster = GeneticAlgorithm(
        pop_size=30, # Tamaño de la poblacion, cantidad de individuos que se generan en cada generacion
        generations=100, #Numero de generaciones hasta que se detiene el algoritmo
        mutation_rate=0.8, #Probabilidad de mutacion, un valor alto introduce diversidad
        patience=7,  #Minimo de iteraciones para poder parar en caso de que el MSE no mejorara
        min_delta=0.003, #Minimo de mejora en el MSE para poder parar, si mejora en este valor se reinciia el contador de paciencia
    )
    ml_task = MachineLearningTask(
        data=df_clustering, #Datos a utilizar
        k=2
    )
    best_solution = ga_cluster.run(ml_task)
