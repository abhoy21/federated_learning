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

#### Code Implementation of model training function:

```python
def train(self, X, y):

        self.mse_values = []
        self.epoch_values = []
        self.batch_msevalues = []
        self.batch_epoch_values = []

        for epoch in range(self.num_epochs):

            indices = np.random.permutation(len(X))
            X_shuffled, y_shuffled = X[indices], y[indices]

            num_batches = len(X) // self.batch_size
            if len(X) % self.batch_size != 0:
                num_batches += 1

            random_batch_index = random.randint(0, num_batches - 1)

            start = random_batch_index * self.batch_size
            end = start + self.batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            y_pred = self.predict(X_batch)
            error = y_batch - y_pred

            mse = np.mean(error ** 2)
            self.batch_msevalues.append(mse)
            self.batch_epoch_values.append(epoch)

            dw = -(2 / self.batch_size) * np.dot(X_batch.T, error)
            db = -(2 / self.batch_size) * np.sum(error)

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            self.mse_values.append(mse)
            self.epoch_values.append(epoch)

        return (
            self.w,
            self.b,
            self.mse_values,
            self.epoch_values,
            self.batch_epoch_values,
            self.batch_msevalues,
        )
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

    def fed_learn(self, X_test, y_test):  # Added X_test, y_test parameters which were missing
        client_data, client_targets = self.client_data_split()
        self.mse_list = []
        self.cycles_list = []
        self.client_contributions = []  # Track client contributions across cycles

        for cycle_idx in range(self.cycles):
            cycle_contributions = []  # Track contributions for this cycle

            # Reset local model to global model parameters
            self.local_model.w = self.global_model.w.copy()  # Use copy() to avoid reference issues
            self.local_model.b = self.global_model.b

            client_weights = []  # Store each client's weights
            client_biases = []   # Store each client's biases

            # First, train all clients and evaluate their contributions
            for client_idx in range(self.n_clients):
                # Train the local model on client data
                self.local_model.train(client_data[client_idx], client_targets[client_idx])

                # Evaluate this client's contribution
                contribution = self.evaluate_client_contribution(
                    self.local_model, client_data[client_idx], client_targets[client_idx])
                cycle_contributions.append(contribution)

                # Store trained weights and biases
                client_weights.append(self.local_model.w.copy())
                client_biases.append(self.local_model.b)

                # Reset local model for next client
                if client_idx < self.n_clients - 1:
                    self.local_model.w = self.global_model.w.copy()
                    self.local_model.b = self.global_model.b

            # Store all client contributions for this cycle
            self.client_contributions.append(cycle_contributions)

            # Now aggregate all client models to update global model
            # Simple average aggregation (can be modified to use contribution scores)
            self.global_model.w = np.mean(client_weights, axis=0)
            self.global_model.b = np.mean(client_biases)

            # Evaluate global model
            mse_global = self.global_model.test(X_test, y_test)
            self.mse_list.append(mse_global)
            self.cycles_list.append(cycle_idx)

            print(f"Cycle {cycle_idx+1}/{self.cycles}, Global MSE: {mse_global:.4f}")
            print(f"Client contributions: {[f'{c:.4f}' for c in cycle_contributions]}")

        return self.global_model, self.mse_list, self.cycles_list, self.client_contributions
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
