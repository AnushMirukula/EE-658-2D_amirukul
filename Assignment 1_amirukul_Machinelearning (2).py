#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Q1(1)
import pandas as pd

# Load the dataset
file_path = 'insurance (2).csv'
insurance_data = pd.read_csv(file_path)

# Display a sample of rows
sample_rows = insurance_data.sample(5)  # You can adjust the number as needed
print("Sample Rows from the Dataset:")
print(sample_rows)


# In[2]:


import pandas as pd

# Assuming insurance_data.csv is your dataset file, adjust the filename accordingly
insurance_data = pd.read_csv('insurance (2).csv')

# Rest of your code
missing_rows = insurance_data.isnull().sum(axis=1)
num_missing_rows = len(missing_rows[missing_rows > 0])

print(f"Number of Rows with Missing Values: {num_missing_rows}")

# Remove rows with missing values
insurance_data_cleaned = insurance_data.dropna()

# Display the shape of the cleaned dataset
print("Shape of the Cleaned Dataset:", insurance_data_cleaned.shape)


# In[10]:


#Q1(3)
import pandas as pd

# One-hot encode 'Region'
insurance_data = pd.get_dummies(insurance_data, columns=['Region'], prefix='Region')

# Binary encode 'Gender' and 'Smoker'
insurance_data['Gender'] = insurance_data['Gender'].map({'male': 0, 'female': 1})
insurance_data['Smoker'] = insurance_data['Smoker'].map({'no': 0, 'yes': 1})

# Display the updated dataset
print("Updated Dataset after Encoding:")
print(insurance_data.head())


# In[11]:


#Q1(4)
from sklearn.preprocessing import MinMaxScaler

# Extract features to be normalized
features_to_normalize = ['Age', 'BMI', 'Children', 'Expenses', 'Gender', 'Smoker', 'Region_northeast', 'Region_northwest', 'Region_southeast', 'Region_southwest']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply Min-Max scaling to selected features
insurance_data[features_to_normalize] = scaler.fit_transform(insurance_data[features_to_normalize])

# Display the updated dataset
print("Updated Dataset after Min-Max Scaling:")
print(insurance_data.head())


# In[12]:


#Q2(1)

# Separate features and target
features = insurance_data.drop('Expenses', axis=1)  # Drop the 'Expenses' column to get features
target = insurance_data['Expenses']  # Select only the 'Expenses' column as the target

# Display the features and target
print("Features:")
print(features.head())

print("\nTarget:")
print(target.head())


# In[13]:


#Q(2)
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Display the shapes of the resulting sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# In[33]:


#Q3(1))
import numpy as np 
from sklearn.metrics import mean_squared_error


# Handle NaN values in features
X_train_filled = np.nan_to_num(X_train)  # Replace NaN with zero and inf with finite numbers

# Handle NaN values in target
y_train_filled = np.nan_to_num(y_train)  # Replace NaN with zero and inf with finite numbers

X_b = np.c_[np.ones((X_train_filled.shape[0], 1)), X_train_filled]
m = X_train_filled.shape[0]  # number of data points
n = X_train_filled.shape[1]  # number of features
alpha = 0.01  # Learning rate
n_iterations = 10000  # Number of iterations
W = np.random.randn(n + 1, 1)  # Weight matrix
loss = []  # Loss value for each iteration

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(W) - y_train_filled.reshape(-1, 1))
    W = W - alpha * gradients
    predictions = X_b.dot(W)

    # Handle NaN and infinite values in predictions
    predictions = np.nan_to_num(predictions)  # Replace NaN with zero and inf with finite numbers

    loss.append(mean_squared_error(y_train_filled, predictions))


# Display the last value of the loss
print("Final Loss:", loss[-1])

# Plotting the loss curve
import matplotlib.pyplot as plt

plt.plot(loss)
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Loss Curve during Training')
plt.show()


# In[34]:


#Q3(2)
# Coefficients and Intercept
intercept = W[0]
coefficients = W[1:]

print("Intercept:", intercept)
print("Coefficients:", coefficients)




# In[35]:


#Q3(3)
import numpy as np
from sklearn.metrics import mean_squared_error

# Function for exponential decay of learning rate
def learning_rate_decay(initial_lr, decay_rate, iteration, decay_step):
    return initial_lr * (decay_rate ** (iteration / decay_step))

# Handle NaN values in features
X_train_filled = np.nan_to_num(X_train)  # Replace NaN with zero and inf with finite numbers

# Handle NaN values in target
y_train_filled = np.nan_to_num(y_train)  # Replace NaN with zero and inf with finite numbers

X_b = np.c_[np.ones((X_train_filled.shape[0], 1)), X_train_filled]
m = X_train_filled.shape[0]  # number of data points
n = X_train_filled.shape[1]  # number of features
initial_learning_rate = 0.01  # Initial learning rate
decay_rate = 0.9  # Decay rate
decay_step = 1000  # Decay step
n_iterations = 10000  # Number of iterations
W = np.random.randn(n + 1, 1)  # Weight matrix
loss = []  # Loss value for each iteration

for iteration in range(n_iterations):
    current_learning_rate = learning_rate_decay(initial_learning_rate, decay_rate, iteration, decay_step)

    gradients = 2/m * X_b.T.dot(X_b.dot(W) - y_train_filled.reshape(-1, 1))
    W = W - current_learning_rate * gradients
    predictions = X_b.dot(W)

    # Handle NaN and infinite values in predictions
    predictions = np.nan_to_num(predictions)  # Replace NaN with zero and inf with finite numbers

    loss.append(mean_squared_error(y_train_filled, predictions))
    
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



# Display the final value of the loss
print("Final Loss:", loss[-1])

# Plotting the loss curve
plt.plot(loss)
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Loss Curve during Training')
plt.show()


# In[36]:


#Q3(4)
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Function for exponential decay of learning rate
def learning_rate_decay(initial_lr, decay_rate, iteration, decay_step):
    return initial_lr * (decay_rate ** (iteration / decay_step))


# Initialize weights for constant learning rate
W_constant_lr = np.random.randn(n + 1, 1)
loss_constant_lr = []

# Initialize weights for decaying learning rate
W_decaying_lr = np.random.randn(n + 1, 1)
loss_decaying_lr = []

for iteration in range(n_iterations):
    # Constant learning rate
    gradients_constant_lr = 2/m * X_b.T.dot(X_b.dot(W_constant_lr) - y_train_filled.reshape(-1, 1))
    W_constant_lr = W_constant_lr - initial_learning_rate * gradients_constant_lr
    predictions_constant_lr = X_b.dot(W_constant_lr)
    loss_constant_lr.append(mean_squared_error(y_train_filled, predictions_constant_lr))

    # Decaying learning rate
    current_learning_rate = learning_rate_decay(initial_learning_rate, decay_rate, iteration, decay_step)
    gradients_decaying_lr = 2/m * X_b.T.dot(X_b.dot(W_decaying_lr) - y_train_filled.reshape(-1, 1))
    W_decaying_lr = W_decaying_lr - current_learning_rate * gradients_decaying_lr
    predictions_decaying_lr = X_b.dot(W_decaying_lr)
    loss_decaying_lr.append(mean_squared_error(y_train_filled, predictions_decaying_lr))

# Plotting the loss curves
plt.figure(figsize=(10, 6))
plt.plot(loss_constant_lr, label='Constant Learning Rate')
plt.plot(loss_decaying_lr, label='Decaying Learning Rate')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Loss Curves for Constant and Decaying Learning Rates')
plt.legend()
plt.show()


# In[38]:


#Q(4)

# Handle NaN values in target for testing dataset
y_test_filled = np.nan_to_num(y_test)

# Compute Mean Absolute Error (MAE) and Mean Squared Error (MSE)
mae = np.mean(np.abs(predictions_test.flatten() - y_test_filled))
mse = np.mean((predictions_test.flatten() - y_test_filled)**2)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)

# Plot a histogram of the error distribution
error_distribution = predictions_test.flatten() - y_test_filled

plt.figure(figsize=(10, 6))
plt.hist(error_distribution, bins=30, edgecolor='black')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Histogram of Prediction Errors')
plt.show()


# In[39]:


#Q(5)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Function for gradient descent
def gradient_descent(X_b, y, learning_rate, n_iterations):
    m = X_b.shape[0]  # number of data points
    n = X_b.shape[1]  # number of features
    W = np.random.randn(n, 1)  # Weight matrix
    loss = []  # Loss value for each iteration

    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(W) - y.reshape(-1, 1))
        W = W - learning_rate * gradients
        predictions = X_b.dot(W)
        loss.append(mean_squared_error(y, predictions))

    return loss

# Generate some synthetic data for demonstration
np.random.seed(42)
X_synthetic = 2 * np.random.rand(100, 1)
y_synthetic = 4 + 3 * X_synthetic + np.random.randn(100, 1)

# Add a bias term to features
X_b_synthetic = np.c_[np.ones((X_synthetic.shape[0], 1)), X_synthetic]

# Varying learning rates for analysis
learning_rates = [0.1, 0.01, 0.001]
n_iterations = 100

plt.figure(figsize=(12, 8))

for learning_rate in learning_rates:
    loss = gradient_descent(X_b_synthetic, y_synthetic.flatten(), learning_rate, n_iterations)
    plt.plot(range(1, n_iterations + 1), loss, label=f'Learning Rate = {learning_rate}')

plt.title('Effect of Learning Rate on Convergence')
plt.xlabel('Number of Iterations')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.show()


# In[40]:


#Q(6)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming X_train and y_train are your training features and target
# Replace X_train and y_train with your actual variable names

# Handle NaN values in features
X_train_filled = np.nan_to_num(X_train)  # Replace NaN with zero and inf with finite numbers

# Handle NaN values in target
y_train_filled = np.nan_to_num(y_train)  # Replace NaN with zero and inf with finite numbers

# Split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X_train_filled, y_train_filled, test_size=0.2, random_state=42)

# Create a Linear Regression model
linear_reg_model = LinearRegression()

# Fit the model on the training data
linear_reg_model.fit(X_train, y_train)

# Predict the target values for the testing data
predictions_test = linear_reg_model.predict(X_test)

# Compute Mean Absolute Error (MAE) and Mean Squared Error (MSE)
mae = mean_absolute_error(y_test, predictions_test)
mse = mean_squared_error(y_test, predictions_test)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)


# In[41]:


#Q(7)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming X_train and y_train are your training features and target
# Replace X_train and y_train with your actual variable names

# Handle NaN values in features
X_train_filled = np.nan_to_num(X_train)  # Replace NaN with zero and inf with finite numbers

# Handle NaN values in target
y_train_filled = np.nan_to_num(y_train)  # Replace NaN with zero and inf with finite numbers

# Split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X_train_filled, y_train_filled, test_size=0.2, random_state=42)

# Add a bias term to the features
X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]

# Use the normal equation to find the regression line directly
theta_normal_eq = np.linalg.inv(X_train_bias.T.dot(X_train_bias)).dot(X_train_bias.T).dot(y_train)

# Add a bias term to the testing features
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Predict the target values for the testing data using the normal equation model
predictions_normal_eq = X_test_bias.dot(theta_normal_eq)

# Compute Mean Absolute Error (MAE) and Mean Squared Error (MSE) for normal equation model
mae_normal_eq = mean_absolute_error(y_test, predictions_normal_eq)
mse_normal_eq = mean_squared_error(y_test, predictions_normal_eq)

print("Mean Absolute Error (MAE) - Normal Equation:", mae_normal_eq)
print("Mean Squared Error (MSE) - Normal Equation:", mse_normal_eq)


# In[66]:


Q(8)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Data Preprocessing
dataset = pd.read_csv('insurance (2).csv')  # Update the file name with your dataset

# Drop rows with missing values
dataset = dataset.dropna()

# Convert categorical variables to numerical using one-hot encoding and binary encoding
dataset = pd.get_dummies(dataset, columns=['Gender', 'Smoker', 'Region'], drop_first=True)

# Normalize features using Min-Max scaling
min_vals = dataset.min()
max_vals = dataset.max()
dataset = (dataset - min_vals) / (max_vals - min_vals)

# Split the data into features and target
X = dataset.drop('Expenses', axis=1)
y = dataset['Expenses']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Descent Implementation
def gradient_descent(X_train, y_train, alpha, n_iterations):
    X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    m = X_train.shape[0]
    n = X_train.shape[1]
    W = np.random.randn(n + 1, 1)
    loss = []

    for iteration in range(n_iterations):
        gradients = 1/m * X_b.T.dot(X_b.dot(W) - y_train.values.reshape(-1, 1))
        W = W - alpha * gradients
        predictions = X_b.dot(W)
        loss.append(mean_squared_error(y_train, predictions))

    return W, loss

# Exponential Decay Learning Rate
def gradient_descent_decay(X_train, y_train, alpha, n_iterations):
    X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    m = X_train.shape[0]
    n = X_train.shape[1]
    W = np.random.randn(n + 1, 1)
    loss = []

    for iteration in range(n_iterations):
        decay_rate = 0.01  # Adjust decay rate as needed
        alpha_decay = alpha / (1 + decay_rate * iteration)
        gradients = 1/m * X_b.T.dot(X_b.dot(W) - y_train.values.reshape(-1, 1))
        W = W - alpha_decay * gradients
        predictions = X_b.dot(W)
        loss.append(mean_squared_error(y_train, predictions))

    return W, loss

# Perform gradient descent
alpha_gradient_descent = 0.01
n_iterations_gradient_descent = 10000
W_gradient_descent, loss_gradient_descent = gradient_descent(X_train, y_train, alpha_gradient_descent, n_iterations_gradient_descent)

# Perform gradient descent with decay
alpha_gradient_descent_decay = 0.01
n_iterations_gradient_descent_decay = 10000
W_gradient_descent_decay, loss_gradient_descent_decay = gradient_descent_decay(X_train, y_train, alpha_gradient_descent_decay, n_iterations_gradient_descent_decay)

# Plot loss values
plt.plot(range(1, n_iterations_gradient_descent + 1), loss_gradient_descent, label='Constant Learning Rate')
plt.plot(range(1, n_iterations_gradient_descent_decay + 1), loss_gradient_descent_decay, label='Decaying Learning Rate')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.title('Gradient Descent Loss')
plt.legend()
plt.show()

# Model Evaluation
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
predictions_test = X_test_b.dot(W_gradient_descent_decay)

mae = mean_absolute_error(y_test, predictions_test)
mse = mean_squared_error(y_test, predictions_test)

print(f'MAE: {mae}')
print(f'MSE: {mse}')


# In[ ]:




