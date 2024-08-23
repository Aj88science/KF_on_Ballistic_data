import numpy as np
import matplotlib.pyplot as plt

# Define the number of data points and the time step
num_samples = 1000
time_step = 0.01  # 10 ms

# Define initial roll, yaw, and pitch values in radians
initial_roll = 0.0
initial_yaw = 0.0
initial_pitch = 0.0

# Create arrays to store the generated data
roll_values = np.zeros(num_samples)
yaw_values = np.zeros(num_samples)
pitch_values = np.zeros(num_samples)

# Create noise parameters (adjust as needed)
roll_noise_stddev = 0.1
yaw_noise_stddev = 0.1
pitch_noise_stddev = 0.1

# Generate synthetic IMU data
for i in range(num_samples):
    # Simulate noise
    roll_noise = np.random.normal(0, roll_noise_stddev)
    yaw_noise = np.random.normal(0, yaw_noise_stddev)
    pitch_noise = np.random.normal(0, pitch_noise_stddev)

    # Update roll, yaw, and pitch with noise
    roll = initial_roll + roll_noise
    yaw = initial_yaw + yaw_noise
    pitch = initial_pitch + pitch_noise

    # Store the values
    roll_values[i] = roll
    yaw_values[i] = yaw
    pitch_values[i] = pitch

    # Update initial values for the next iteration (e.g., to simulate motion)
    initial_roll = roll
    initial_yaw = yaw
    initial_pitch = pitch

# Kalman Filter Initialization
# Initial State
x = np.zeros((2, 1))  # [angle; bias]
# Initial Estimation Error Covariance
P = np.eye(2)
# Initial Process Noise Covariance
Q = np.array([[0.0001, 0], [0, 0.0001]])
# Measurement Noise Covariance
R = np.array([[roll_noise_stddev**2]])

# State Transition Matrix
A = np.array([[1, -time_step], [0, 1]])
# Measurement Matrix
H = np.array([[1, 0]])

# Smoothed Data
smoothed_roll_values = np.zeros(num_samples)

# Define Genetic Algorithm parameters
population_size = 50
num_generations = 10
mutation_rate = 0.1

# Fitness function
def evaluate_fitness(params):
    # Kalman Filter Loop
    x = np.zeros((2, 1))
    P = np.eye(2)
    smoothed_roll_values = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Prediction
        x = np.dot(A, x)
        P = np.dot(np.dot(A, P), A.T) + np.array([[params[0], 0], [0, params[1]]])  # Process Noise Covariance matrix
        # Update
        S = np.dot(np.dot(H, P), H.T) + np.array([[params[2]]])  # Measurement Noise Covariance
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
        y = roll_values[i] - np.dot(H, x)
        x = x + np.dot(K, y)
        P = P - np.dot(np.dot(K, H), P)
        # Store smoothed data
        smoothed_roll_values[i] = x[0, 0]

    # Fitness: Mean Squared Error between smoothed_roll_values and roll_values
    return np.mean((smoothed_roll_values - roll_values)**2)

# Genetic Algorithm
best_fitness = float('inf')
best_params = None

for _ in range(num_generations):
    # Generate new population
    population = np.random.uniform(0.01, 0.1, size=(population_size, 3))  # Parameters: Process Noise Covariance, Measurement Noise Covariance

    # Evaluate fitness for each individual
    fitness_scores = [evaluate_fitness(individual) for individual in population]

    # Select best individual
    best_index = np.argmin(fitness_scores)
    if fitness_scores[best_index] < best_fitness:
        best_fitness = fitness_scores[best_index]
        best_params = population[best_index]

    # Roulette wheel selection
    fitness_sum = np.sum(fitness_scores)
    probabilities = [score / fitness_sum for score in fitness_scores]
    selected_indices = np.random.choice(range(population_size), size=population_size, p=probabilities)

    # Crossover
    crossover_point = np.random.randint(1, 3, size=population_size)
    offspring = np.zeros_like(population)
    for i, point in enumerate(crossover_point):
        parent1, parent2 = population[selected_indices[i]], population[selected_indices[(i+1) % population_size]]
        offspring[i] = np.concatenate((parent1[:point], parent2[point:]))

    # Mutation
    mask = np.random.rand(population_size, 3) < mutation_rate
    mutation_noise = np.random.normal(0, 0.01, size=(population_size, 3))
    offspring[mask] += mutation_noise[mask]
   
# Re-evaluate best individual
best_fitness = evaluate_fitness(best_params)

# Update Kalman Filter with best parameters
x = np.zeros((2, 1))
P = np.eye(2)
for i in range(num_samples):
    # Prediction
    x = np.dot(A, x)
    P = np.dot(np.dot(A, P), A.T) + np.array([[best_params[0], 0], [0, best_params[1]]])  # Process Noise Covariance matrix
    # Update
    S = np.dot(np.dot(H, P), H.T) + np.array([[best_params[2]]])  # Measurement Noise Covariance
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    y = roll_values[i] - np.dot(H, x)
    x = x + np.dot(K, y)
    P = P - np.dot(np.dot(K, H), P)
    # Store smoothed data
    smoothed_roll_values[i] = x[0, 0]

# Plotting
time_values = np.arange(0, num_samples * time_step, time_step)
plt.figure(figsize=(12, 6))
plt.plot(time_values, roll_values, label='Original Roll (rad)', alpha=0.5)
plt.plot(time_values, smoothed_roll_values, label='Smoothed Roll (rad) - KF + GA', color='red')
plt.title('IMU Roll with Kalman Filter Smoothing (Genetic Algorithm Tuned)')
plt.xlabel('Time (s)')
plt.ylabel('Roll (rad)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("Best parameters (Process Noise Covariance, Measurement Noise Covariance):", best_params)
print("Best fitness (MSE):", best_fitness)
