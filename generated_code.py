import numpy as np

def perceptron_and(x1, x2, weight1, weight2, bias):
    """
    Perceptron implementation of the AND operation.

    Args:
        x1: First input value.
        x2: Second input value.
        weight1: Weight for the first input.
        weight2: Weight for the second input.
        bias: Bias value.

    Returns:
        The output of the AND operation (0 or 1).
    """
    # Calculate the weighted sum of inputs and bias
    weighted_sum = (weight1 * x1) + (weight2 * x2) + bias

    # Apply the activation function (step function)
    if weighted_sum >= 0:
        return 1
    else:
        return 0

# Example usage:
weight1 = 1
weight2 = 1
bias = -1.5  # 중요: bias 값을 조정하여 AND 연산을 구현합니다.

print(f"AND(0, 0) = {perceptron_and(0, 0, weight1, weight2, bias)}")  # Output: 0
print(f"AND(0, 1) = {perceptron_and(0, 1, weight1, weight2, bias)}")  # Output: 0
print(f"AND(1, 0) = {perceptron_and(1, 0, weight1, weight2, bias)}")  # Output: 0
print(f"AND(1, 1) = {perceptron_and(1, 1, weight1, weight2, bias)}")  # Output: 1