import numpy as np
import matplotlib.pyplot as plt

# Define the number of data points and the time step
num_samples = 1000
time_step = 0.001  # 1 ms

# Define initial roll, yaw, and pitch values in radians
initial_roll = 0.0
initial_yaw = 0.0
initial_pitch = 0.0

# Create arrays to store the generated data
actual_roll_values = np.zeros(num_samples)
roll_values = np.zeros(num_samples)
yaw_values = np.zeros(num_samples)
pitch_values = np.zeros(num_samples)

# Create noise parameters (adjust as needed)
roll_noise_stddev = 0.1
yaw_noise_stddev = 0.1
pitch_noise_stddev = 0.1

# Generate synthetic IMU data
# Generate arrays of random samples for roll_noise, yaw_noise, and pitch_noise
roll_noise = np.random.normal(0, roll_noise_stddev, num_samples)
yaw_noise = np.random.normal(0, yaw_noise_stddev, num_samples)
pitch_noise = np.random.normal(0, pitch_noise_stddev, num_samples)

for i in range(num_samples):
    # Update roll, yaw, and pitch with noise
    roll = initial_roll + roll_noise[i]
    yaw = initial_yaw + yaw_noise[i]
    pitch = initial_pitch + pitch_noise[i]

    # Store the values
    roll_values[i] = roll
    yaw_values[i] = yaw
    pitch_values[i] = pitch

    # Store the actual (noiseless) roll values
    actual_roll_values[i] = initial_roll

    # Update initial values for the next iteration (e.g., to simulate motion)
    initial_roll = roll
    initial_yaw = yaw
    initial_pitch = pitch

# Create time values
time_values = np.arange(0, num_samples * time_step, time_step)
# Define initial state estimate (e.g., zero) and covariance
state_estimate = 0.0
state_covariance = 5.3

# Define process noise and measurement noise (adjust as needed)
process_noise = 0.5
measurement_noise = 1.0

# Define state transition matrix (F) and measurement matrix (H)
F = 42.04  # For simplicity, assume a constant velocity model
H = 1.3  # Measuring the roll directly

# Initialize Kalman filter
def init_kalman_filter():
    global state_estimate, state_covariance
    state_estimate = 0.0
    state_covariance = 1.3

# Predict step
def predict(control_input=0.5):
    global state_estimate, state_covariance

    # Prediction step
    state_estimate = F * state_estimate + control_input
    state_covariance = F * state_covariance * F + process_noise

# Update step
def update(measurement):
    global state_estimate, state_covariance

    # Calculate Kalman gain
    kalman_gain = state_covariance * H / (H * state_covariance * H + measurement_noise)

    # Update state estimate
    state_estimate = state_estimate + kalman_gain * (measurement - H * state_estimate)

    # Update state covariance
    state_covariance = (1 - kalman_gain * H) * state_covariance

# Example usage
init_kalman_filter()
filtered_roll_values = []

for roll_measurement in roll_values:
    predict()  # Optional control input can be added here
    update(roll_measurement)
    filtered_roll_values.append(state_estimate)



# Plot the roll, yaw, and pitch data with and without noise
plt.figure(figsize=(12, 6))
plt.plot(time_values, roll_values, label='Initial Roll (with noise)', color='blue')
plt.plot(time_values, actual_roll_values, label='Actual Roll (noiseless)', color='green')
plt.plot(time_values, filtered_roll_values, label='Filtered Roll (rad)', color='red')
plt.title('IMU Roll Data, Actual Roll, and Kalman Filtered Roll')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend()
plt.show()
