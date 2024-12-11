import numpy as np
import matplotlib.pyplot as plt

# Define training data (features and labels)
x_train = np.array([0., 1, 2, 3, 4, 5])  # Input feature values
y_train = np.array([0, 0, 0, 1, 1, 1])  # Corresponding binary labels

# Initialize weight and bias for the sigmoid regression model
w_in = 0.1  # Initial weight
b_in = 0    # Initial bias

def model(x, w, b):
    """
    Computes the linear combination of input features, weights, and bias.

    Parameters:
    x (array): Input features
    w (float): Weight
    b (float): Bias

    Returns:
    array: Linear combination (z)
    """
    z = np.dot(x, w) + b
    return z

def sigmoid(z):
    """
    Applies the sigmoid function to map input z to a range (0, 1).

    Parameters:
    z (array): Linear combination from the model

    Returns:
    array: Sigmoid output representing probabilities
    """
    y = 1 / (1 + np.exp(-z))
    return y

# Compute the linear combination (z) using the model
f_wb = model(x_train, w_in, b_in)

# Apply the sigmoid (sigmoid) function to calculate probabilities
fina = sigmoid(f_wb)

# Print the predicted probabilities
print(fina)

# Plot the training data points
plt.scatter(x_train, y_train, color='red', label='Training data')

# Plot the sigmoid regression prediction curve
plt.plot(x_train, fina, label='sigmoid Model', color='blue')

# Label the axes
plt.xlabel('x')  # Label for x-axis
plt.ylabel('Predicted probability')  # Label for y-axis

# Add a title to the plot
plt.title('sigmoid Regression Model')

# Add a legend to differentiate between data points and model
plt.legend()

# Display the plot
plt.show()
