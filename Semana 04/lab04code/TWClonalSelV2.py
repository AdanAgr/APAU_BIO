import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

##############################################################################
# (A) Load time-series data from CSV (normal vs. anomalies)
##############################################################################

def load_timeseries_from_csv(filepath, window_size=20, step=20):
    data = pd.read_csv(filepath)
    T = data['value'].values  # Asumiendo que la columna de interés se llama 'value'
    
    # Definir intervalos de anomalías basados en el CSV
    anomaly_intervals = [(100, 120), (250, 270)]  # Ajustar según sea necesario

    # Slice into windows
    n_points = len(T)
    window_starts = range(0, n_points - window_size + 1, step)
    X, y = [], []
    for ws in window_starts:
        we = ws + window_size
        window_data = T[ws:we]
        # label=1 if overlaps any anomaly interval
        label = 0
        for (a_start, a_end) in anomaly_intervals:
            if not (we <= a_start or ws >= a_end):
                label = 1
                break
        X.append(window_data)
        y.append(label)

    X = np.array(X)
    y = np.array(y, dtype=int)
    return T, X, y, list(window_starts)

def plot_timeseries_with_windows(
    T,
    anomaly_intervals,
    window_size,
    window_starts,
    y,
    title="Time Series with Windows"
):
    n_points = len(T)
    plt.figure(figsize=(12, 4))

    # Plot entire series in blue
    plt.plot(np.arange(n_points), T, color='blue', lw=1)

    # Overwrite anomaly intervals in red
    for (a_start, a_end) in anomaly_intervals:
        plt.plot(np.arange(a_start, a_end), T[a_start:a_end], color='red', lw=1)

    # Draw vertical spans for each window
    for i, ws in enumerate(window_starts):
        we = ws + window_size
        label = y[i]
        color = 'orange' if label==1 else 'green'
        plt.axvspan(ws, we, color=color, alpha=0.1)

    plt.title(title)
    plt.xlabel("Time Index")
    plt.ylabel("Signal Amplitude")
    plt.xlim(0, n_points)
    plt.legend([
        "Full Series (blue=normal, red=anomaly)",
        "Window shading (green=normal, orange=anomaly)"
    ])
    plt.show()

##############################################################################
# (B) Clonal Selection AIS Class
##############################################################################
class ClonalSelectionAIS:
    """
    A simplified clonal selection algorithm for one-class anomaly detection.
    """
    def __init__(self,
                 pop_size=30,
                 clone_factor=5,
                 beta=1.0,
                 mutation_std=0.1,
                 max_gens=10,
                 diversity_rate=0.1,
                 random_seed=123):
        self.pop_size = pop_size
        self.clone_factor = clone_factor
        self.beta = beta
        self.mutation_std = mutation_std
        self.max_gens = max_gens
        self.diversity_rate = diversity_rate
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.population_ = None
        self.threshold_ = None
        self.loss_history_ = []

    def _init_population(self, X):
        mins = X.min(axis=0) - 0.5
        maxs = X.max(axis=0) + 0.5
        n_features = X.shape[1]
        self.population_ = np.random.uniform(mins, maxs,
                                             size=(self.pop_size, n_features))

    def _affinity(self, antibody, X_normal):
        dists = np.linalg.norm(X_normal - antibody, axis=1)
        return 1.0 / (1.0 + dists.mean())

    def _evaluate_pop(self, X_normal):
        return np.array([self._affinity(ab, X_normal) for ab in self.population_])

    def _clone_and_mutate(self, population, affs):
        sorted_idx = np.argsort(affs)[::-1]
        clones_list = []
        eps = 1e-9
        for rank, idx in enumerate(sorted_idx):
            parent_aff = affs[idx]
            parent = population[idx].copy()
            clone_count = int(self.clone_factor * (rank + 1))
            for _ in range(clone_count):
                clone = parent.copy()
                mut_rate = self.beta / (parent_aff + eps)
                noise = np.random.normal(0, self.mutation_std * mut_rate,
                                         size=clone.shape)
                clone += noise
                clones_list.append(clone)
        if len(clones_list) == 0:
            return population
        return np.array(clones_list)

    def fit(self, X_normal, X_val=None, y_val=None):
        self._init_population(X_normal)

        for gen in range(self.max_gens):
            affs = self._evaluate_pop(X_normal)
            clones = self._clone_and_mutate(self.population_, affs)
            clone_affs = [self._affinity(c, X_normal) for c in clones]
            clone_affs = np.array(clone_affs)

            combined = np.vstack([self.population_, clones])
            combined_affs = np.concatenate([affs, clone_affs])

            best_idx = np.argsort(combined_affs)[::-1][:self.pop_size]
            self.population_ = combined[best_idx]

            self.threshold_ = self._compute_threshold(X_normal)

            loss = None
            if X_val is not None and y_val is not None:
                y_pred = self.predict(X_val)
                loss = np.mean(y_pred != y_val)
                self.loss_history_.append(loss)
            else:
                y_pred_norm = self.predict(X_normal)
                loss = np.mean(y_pred_norm == 1)
                self.loss_history_.append(loss)

    def _compute_threshold(self, X_normal):
        min_dists = []
        for x in X_normal:
            dists = np.linalg.norm(self.population_ - x, axis=1)
            min_dists.append(dists.min())
        return max(min_dists)

    def predict(self, X):
        if self.threshold_ is None:
            raise ValueError("Model not fitted, threshold is None.")
        labels = np.zeros(len(X), dtype=int)
        for i, x in enumerate(X):
            dists = np.linalg.norm(self.population_ - x, axis=1)
            if dists.min() > self.threshold_:
                labels[i] = 1
        return labels

##############################################################################
# (C) Demo: Train & Evaluate
##############################################################################
if __name__ == "__main__":

    train_filepath = "./archive/artificialNoAnomaly/artificialNoAnomaly/art_flatline.csv"
    test_filepath = "./archive/artificialWithAnomaly/artificialWithAnomaly/art_daily_flatmiddle.csv"

    # Load train and test datasets
    _, X_train, y_train, _ = load_timeseries_from_csv(filepath=train_filepath, window_size=100, step=20)
    _, X_test, y_test, _ = load_timeseries_from_csv(filepath=test_filepath, window_size=100, step=20)

    # Train on normal samples
    X_train_normal = X_train[y_train == 0]

    ais = ClonalSelectionAIS(
        pop_size=30,
        clone_factor=5,
        beta=1.0,
        mutation_std=0.1,
        max_gens=10,
        diversity_rate=0.1,
        random_seed=123
    )

    ais.fit(X_train_normal, X_val=X_test, y_val=y_test)

    # Plot loss history
    plt.figure()
    plt.plot(ais.loss_history_, marker='o')
    plt.title("Clonal AIS: Loss History")
    plt.xlabel("Generation")
    plt.ylabel("Loss (Test Misclassification)")
    plt.grid(True)
    plt.show()

    # Evaluate on test set
    y_pred = ais.predict(X_test)

    labels = [0, 1]  # 0: Normal, 1: Anomaly
    print("Confusion Matrix (Test):")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(cm)

    target_names = ["Normal", "Anomaly"]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))