import numpy as np
import matplotlib.pyplot as plt
import time
import random
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa

# Simple Homomorphic Encryption (HE) class
class SimpleHE:
    def __init__(self):
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
        self.public_key = self.private_key.public_key()

    def encrypt(self, value):
        return value + 1000  # Dummy encryption by adding 1000

    def decrypt(self, encrypted_value):
        return encrypted_value - 1000  # Dummy decryption by subtracting 1000

    def homomorphic_add(self, value1, value2):
        return value1 + value2

# Gradient Clipping
def gradient_clipping(update, threshold=1.0):
    return np.clip(update, -threshold, threshold)

# Top-k Sparsification (model compression)
def top_k_sparsification(update, k=3):
    indices = np.argsort(np.abs(update))[-k:]  # Keep the top k elements
    sparse_update = np.zeros_like(update)
    sparse_update[indices] = update[indices]
    return sparse_update

# Adaptive Noise Addition
def add_adaptive_noise(value, epsilon, convergence_ratio):
    noise_scale = 1 / (epsilon * (1 + convergence_ratio))  # Adjust noise based on convergence ratio
    noise = np.random.laplace(loc=0, scale=noise_scale, size=value.shape)
    return value + noise

# Federated Learning with Efficiency Improvements
class FederatedLearningEfficient:
    def __init__(self, num_clients, epsilon, k, subset_size, clip_threshold, model_size):
        self.num_clients = num_clients
        self.epsilon = epsilon
        self.k = k  # Top-k compression
        self.subset_size = subset_size  # Number of clients selected in each round
        self.clip_threshold = clip_threshold  # Gradient clipping threshold
        self.global_model = np.zeros(model_size)  # Initialize global model as an array (e.g., vector or gradient)
        self.he = SimpleHE()
        self.model_updates = []
        self.communication_costs = []
        self.accuracies = []
        self.training_time = []

    def local_update(self, client_data, convergence_ratio):
        local_model_update = np.mean(client_data, axis=0)  # Model update is an array (e.g., gradient vector)
        clipped_update = gradient_clipping(local_model_update, self.clip_threshold)
        compressed_update = top_k_sparsification(clipped_update, self.k)
        noisy_update = add_adaptive_noise(compressed_update, self.epsilon, convergence_ratio)
        encrypted_update = self.he.encrypt(noisy_update)
        return encrypted_update

    def aggregate_updates(self, encrypted_updates):
        aggregated_update = np.zeros_like(encrypted_updates[0])
        for encrypted_update in encrypted_updates:
            aggregated_update = self.he.homomorphic_add(aggregated_update, encrypted_update)
        return self.he.decrypt(aggregated_update)

    def train(self, client_data_list, round_number):
        start_time = time.time()

        # Select a subset of clients
        selected_clients = random.sample(range(self.num_clients), self.subset_size)
        encrypted_updates = []
        convergence_ratio = round_number / 10  # Simulate convergence ratio (closer to 1 as rounds increase)

        for client_id in selected_clients:
            client_data = client_data_list[client_id]
            encrypted_update = self.local_update(client_data, convergence_ratio)
            encrypted_updates.append(encrypted_update)

        global_update = self.aggregate_updates(encrypted_updates)
        self.global_model += global_update
        self.model_updates.append(self.global_model.copy())  # Save a copy of the global model
        self.communication_costs.append(len(encrypted_updates) * 8 * len(self.global_model))  # 8 bytes per encrypted float

        self.training_time.append(time.time() - start_time)
        self.accuracies.append(self.calculate_accuracy())  # Simulate accuracy

    def calculate_accuracy(self):
        # Dummy accuracy calculation based on the global model value
        return 1 - np.linalg.norm(self.global_model - np.ones_like(self.global_model)) / np.linalg.norm(np.ones_like(self.global_model))

# Helper function to pad results
def pad_results(shorter_list, target_length):
    return shorter_list + [shorter_list[-1]] * (target_length - len(shorter_list))

# Simulate the training process with efficiency improvements
def simulate_efficient_training(num_clients=5, epsilon=0.5, rounds=10, k=3, subset_size=3, clip_threshold=1.0, model_size=10):
    client_data_list = [np.random.rand(10, model_size) * (i + 1) for i in range(num_clients)]  # Create array of model updates

    efficient_fl = FederatedLearningEfficient(num_clients, epsilon, k, subset_size, clip_threshold, model_size)

    for round in range(rounds):
        efficient_fl.train(client_data_list, round + 1)

    max_len = len(efficient_fl.model_updates)

    model_updates = pad_results(efficient_fl.model_updates, max_len)
    comm_costs = pad_results(efficient_fl.communication_costs, max_len)
    accuracies = pad_results(efficient_fl.accuracies, max_len)
    times = pad_results(efficient_fl.training_time, max_len)

    return model_updates, comm_costs, accuracies, times

# Plot results
def plot_efficient_results(model_updates, comm_costs, accuracies, times):
    rounds = np.arange(1, len(model_updates) + 1)

    plt.figure(figsize=(14, 10))

    # Model Updates Comparison
    plt.subplot(2, 2, 1)
    plt.plot(rounds, [np.mean(mu) for mu in model_updates], label='Model Updates', marker='o')
    plt.title('Model Updates')
    plt.xlabel('Rounds')
    plt.ylabel('Global Model Value')
    plt.legend()
    plt.grid()

    # Communication Costs
    plt.subplot(2, 2, 2)
    plt.plot(rounds, comm_costs, label='Comm Cost', marker='o')
    plt.title('Communication Costs')
    plt.xlabel('Rounds')
    plt.ylabel('Communication Cost (Bytes)')
    plt.legend()
    plt.grid()

    # Accuracy Comparison
    plt.subplot(2, 2, 3)
    plt.plot(rounds, accuracies, label='Accuracy', marker='o')
    plt.title('Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # Training Time Comparison
    plt.subplot(2, 2, 4)
    plt.plot(rounds, times, label='Training Time (s)', marker='o')
    plt.title('Training Time')
    plt.xlabel('Rounds')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Run simulation and plot results
efficient_results = simulate_efficient_training()
plot_efficient_results(*efficient_results)
