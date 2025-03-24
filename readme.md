# Federated Learning with Regressor Model

## Introduction

This project demonstrates a Federated Learning framework using a simple linear regression model. The core components of the project include:

- **Regressor Class**: Implements a linear regression model with batch gradient descent.
- **Federated Class**: Simulates federated learning by training multiple client models and aggregating updates to a global model.

The workflow includes dataset loading, client data partitioning, local training, aggregation of updates, and global model evaluation.

## Workflow Overview

### 1. Dataset Loading and Preprocessing

Before training begins, we load and preprocess the dataset, splitting it into features (`X`) and target values (`y`). The dataset is then divided among clients for federated learning.

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Simulated dataset
X = np.random.rand(1000, 5)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(1000)  # Linear relationship

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. Regressor Class (Local Model)

The `Regressor` class implements a linear regression model trained using Mean Squared Error (MSE) minimization via batch gradient descent.

#### Key Features:

- Supports batch processing for efficient training.
- Updates weights (`w`) and bias (`b`) using gradient descent.
- Tracks training progress with MSE values.

#### Code Implementation:

```python
class Regressor:
    def __init__(self, w, b, batch_size, learning_rate, num_epochs):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.w = w
        self.b = b

    def train(self, X, y):
        for epoch in range(self.num_epochs):
            y_pred = self.predict(X)
            error = y - y_pred

            # Compute gradients
            dw = -(2 / len(X)) * np.dot(X.T, error)
            db = -(2 / len(X)) * np.sum(error)

            # Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b
```

### 3. Federated Learning Framework

The `Federated` class simulates federated learning, where multiple clients train their local models and contribute updates to the global model.

#### Key Features:

- Splits dataset among `n_clients`.
- Each client trains a local model and submits updates.
- Aggregates updates using simple averaging.
- Evaluates global model performance after each cycle.

#### Code Implementation:

```python
class Federated:
    def __init__(self, global_model, local_model, cycles, n_clients, X, y):
        self.global_model = global_model
        self.local_model = local_model
        self.cycles = cycles
        self.n_clients = n_clients
        self.X = X
        self.y = y

    def fed_learn(self, X_test, y_test):
        client_data, client_targets = np.array_split(self.X, self.n_clients), np.array_split(self.y, self.n_clients)

        for cycle in range(self.cycles):
            client_weights, client_biases = [], []

            for client_idx in range(self.n_clients):
                self.local_model.train(client_data[client_idx], client_targets[client_idx])
                client_weights.append(self.local_model.w.copy())
                client_biases.append(self.local_model.b)

            # Aggregate updates (average model parameters)
            self.global_model.w = np.mean(client_weights, axis=0)
            self.global_model.b = np.mean(client_biases)

            mse_global = self.global_model.test(X_test, y_test)
            print(f"Cycle {cycle+1}, Global MSE: {mse_global:.4f}")
```

### 4. Model Training and Evaluation

To train the global model using federated learning, instantiate the models and call `fed_learn`:

```python
# Initialize models
initial_weights = np.zeros(X_train.shape[1])
global_model = Regressor(w=initial_weights, b=0, batch_size=10, learning_rate=0.01, num_epochs=10)
local_model = Regressor(w=initial_weights, b=0, batch_size=10, learning_rate=0.01, num_epochs=5)

# Federated training
federated_system = Federated(global_model, local_model, cycles=5, n_clients=4, X=X_train, y=y_train)
federated_system.fed_learn(X_test, y_test)
```

### 5. Visualizing Client Contributions

The federated framework also tracks client contributions, which can be visualized as follows:

```python
federated_system.visualize_client_contributions()
```

## Summary

This project demonstrates how a linear regression model can be trained using federated learning. The `Regressor` class handles local training, while the `Federated` class simulates decentralized training with multiple clients, aggregating updates to improve the global model over multiple cycles.

### Future Improvements:

- Implement weighted aggregation based on client performance.
- Introduce differential privacy for enhanced security.
- Support more complex models such as neural networks.

This project provides a strong foundation for federated learning applications in real-world scenarios.
