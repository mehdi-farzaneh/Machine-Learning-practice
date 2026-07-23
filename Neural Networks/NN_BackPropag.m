%% Neural Network Learning with back-propagation
clear ; close all; clc

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10 ("0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
fprintf('Loading and Visualizing Data ...\n')
load('data2.mat');
m = size(X, 1);
% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
NN.displayData(X(sel, :));
fprintf('Press enter to continue.\n');
pause;

%% ================ Part 2: Loading Parameters ================
fprintf('\nLoading Saved Neural Network Parameters ...\n')
load('weights2.mat');
nn_params = [Theta1(:) ; Theta2(:)];

lambda = 0; % feedforward cost *without* regularization first
J = NN.NNcostfun(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
                   
lambda = 1; % feedforward cost *with* regularization first
J = NN.NNcostfun(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

%% ================ Part 6: Initializing Pameters ================
fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = NN.randInitWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = NN.randInitWeights(hidden_layer_size, num_labels);
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

NN.checkNNGrad; % Backpropagation *without* regularization 
lambda = 3;
NN.checkNNGrad(lambda); % Backpropagation *with* regularization 

debug_J  = NN.NNcostfun(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);
fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f '], ...
          lambda, debug_J);
fprintf('Press enter to continue.\n');
pause;

%% =================== Part 8: Training NN ===================
fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 50);
lambda = 1; % try different values 
costFunction = @(p) NN.NNcostfun(p, input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
% costFunction takes in only the neural network parameters
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
fprintf('Press enter to continue.\n');
pause;

%% ================= Part 9: Visualize Weights =================
fprintf('\nVisualizing Neural Network... \n')
NN.displayData(Theta1(:, 2:end));
fprintf('\nPress enter to continue.\n');
pause;

%% ================= Part 10: Implement Predict =================
pred = NN.predict2(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

