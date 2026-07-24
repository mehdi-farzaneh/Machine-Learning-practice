function g = sigmoid(z)
% Sigmoid activation function: maps any real-valued input to the range (0, 1).

% Compute the sigmoid element-wise for the input array or matrix.
g = 1 ./ (1 + exp(-z));
end
