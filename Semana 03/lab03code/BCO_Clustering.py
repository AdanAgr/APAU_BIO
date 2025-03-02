import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode

class BeeColonyClustering:

    def __init__(
        self,
        data,
        num_clusters=3,
        num_bees=30,
        num_employed=15,
        num_scouts=5,
        generations=50,
        top_solutions=5,
        seed=None
    ):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.data = np.array(data)
        self.num_samples, self.num_features = self.data.shape
        self.num_clusters = num_clusters

        self.num_bees = num_bees
        self.num_employed = num_employed
        self.num_scouts = num_scouts
        self.generations = generations
        self.top_solutions = top_solutions

        self.data_min = self.data.min(axis=0)
        self.data_max = self.data.max(axis=0)

        self.positions = np.random.uniform(
            low=self.data_min, high=self.data_max,
            size=(self.num_bees, self.num_clusters, self.num_features)
        )

        self.best_solution = None
        self.best_fitness = np.inf

    def fitness(self, centroids):
        distances = cdist(self.data, centroids, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2)

    def _bound_solution(self, solution):
        return np.clip(solution, self.data_min, self.data_max)

    def _local_search(self, current_position):
        direction = (self.best_solution - current_position)
        step = np.random.uniform(-0.1, 0.1, current_position.shape) * direction
        new_position = current_position + step
        return self._bound_solution(new_position)

    def update_positions(self):
        for i in range(self.num_employed):
            new_position = self._local_search(self.positions[i])
            new_fitness = self.fitness(new_position)
            if new_fitness < self.fitness(self.positions[i]):
                self.positions[i] = new_position

        fitness_values = np.array([self.fitness(pos) for pos in self.positions])
        probabilities = np.exp(-fitness_values)
        probabilities /= probabilities.sum()

        selected_indices = np.random.choice(
            range(self.num_bees), size=self.num_employed, p=probabilities
        )
        for i, idx in enumerate(selected_indices):
            candidate = self._local_search(self.positions[idx])
            if self.fitness(candidate) < self.fitness(self.positions[self.num_employed + i]):
                self.positions[self.num_employed + i] = candidate

        for i in range(self.num_scouts):
            random_explore = self.best_solution + np.random.uniform(-1.0, 1.0, self.best_solution.shape)
            self.positions[-(i+1)] = self._bound_solution(random_explore)

    def _finalize_centroids(self, top_positions):
        all_centroids = top_positions.reshape(-1, self.num_features)

        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=42)
        kmeans.fit(all_centroids)
        return kmeans.cluster_centers_

    def run(self):
        initial_fitnesses = [self.fitness(pos) for pos in self.positions]
        best_idx = np.argmin(initial_fitnesses)
        self.best_fitness = initial_fitnesses[best_idx]
        self.best_solution = self.positions[best_idx].copy()

        for gen in range(self.generations):
            self.update_positions()

            fitnesses = np.array([self.fitness(pos) for pos in self.positions])
            current_best_idx = np.argmin(fitnesses)
            current_best_fit = fitnesses[current_best_idx]
            if current_best_fit < self.best_fitness:
                self.best_fitness = current_best_fit
                self.best_solution = self.positions[current_best_idx].copy()

            print(f"Gen {gen+1}/{self.generations} | Best Fitness: {self.best_fitness:.2f}")

        fitnesses = np.array([self.fitness(pos) for pos in self.positions])
        sorted_indices = np.argsort(fitnesses)
        top_indices = sorted_indices[:self.top_solutions]
        top_positions = self.positions[top_indices]

        final_centroids = self._finalize_centroids(top_positions)
        self.best_solution = final_centroids
        self.best_fitness = self.fitness(final_centroids)

        print(f"\nBCO Finished | Final Best Fitness: {self.best_fitness:.2f}")
        return self.best_solution

    def assign_test_data(self, test_data):
        test_distances = cdist(test_data, self.best_solution, metric='euclidean')
        test_labels = np.argmin(test_distances, axis=1)
        return test_labels

    def plot_clusters(self, train_data):
        train_labels = np.argmin(cdist(train_data, self.best_solution, metric='euclidean'), axis=1)

        plt.figure(figsize=(8, 6))
        plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap='tab10', edgecolors='k', marker='o')
        plt.scatter(self.best_solution[:, 0], self.best_solution[:, 1], c='red', marker='X', s=200, label='Centroids')
        plt.title("BCO Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        # Add hyperparameters to plot
        hyperparameters = (
            f"Num Clusters: {self.num_clusters}\n"
            f"Num Bees: {self.num_bees}\n"
            f"Num Employed: {self.num_employed}\n"
            f"Num Scouts: {self.num_scouts}\n"
            f"Generations: {self.generations}\n"
            f"Top Solutions: {self.top_solutions}"
        )
        plt.gcf().text(0.95, 0.05, hyperparameters, fontsize=12, ha='right', bbox=dict(facecolor='red', alpha=0.5))
        
        plt.legend()
        plt.show()

def main():
    print("=== Bee Colony Optimization Clustering ===\n")
    # Load dataset
    file_path = "Mall_Customers.csv"  
    data = pd.read_csv(file_path)

    # Select relevant columns
    numeric_columns = ["Annual Income (k$)", "Spending Score (1-100)"]
    numeric_data = data[numeric_columns]

    # Normalize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Run BCO clustering
    bco_clustering = BeeColonyClustering(
        data=scaled_data,
        num_clusters=4,
        num_bees=30,
        num_employed=15,
        num_scouts=5,
        generations=50,
        top_solutions=5,
        seed=42
    )

    bco_clustering.run()
    # Plot results
    bco_clustering.plot_clusters(scaled_data)

if __name__ == "__main__":
    main()
