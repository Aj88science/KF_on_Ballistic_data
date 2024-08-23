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

# Kalman Filter Loop
for i in range(num_samples):
    # Prediction
    x = np.dot(A, x)
    P = np.dot(np.dot(A, P), A.T) + Q

    # Update
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    y = roll_values[i] - np.dot(H, x)
    x = x + np.dot(K, y)
    P = P - np.dot(np.dot(K, H), P)

    # Store smoothed data
    smoothed_roll_values[i] = x[0, 0]

# Create time values
time_values = np.arange(0, num_samples * time_step, time_step)

# Plot the original and smoothed roll data
plt.figure(figsize=(12, 6))
plt.plot(time_values, roll_values, label='Original Roll (rad)', alpha=0.5)
plt.plot(time_values, smoothed_roll_values, label='Smoothed Roll (rad)', color='red')
plt.title('IMU Roll with Kalman Filter Smoothing')
plt.xlabel('Time (s)')
plt.ylabel('Roll (rad)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
