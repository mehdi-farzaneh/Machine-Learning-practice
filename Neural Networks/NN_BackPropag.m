%% Neural network learning with backpropagation
clear; close all; clc

inputLayerSize = 400;   % 20x20 input images of digits
hiddenLayerSize = 25;   % 25 hidden units
numLabels = 10;         % 10 labels, from 1 to 10 ("0" is labeled as 10)

%% =========== Part 1: Load and visualize the data =============
fprintf('Loading and visualizing data...\n');
load('data2.mat');

m = size(X, 1);

% Randomly select 100 examples to display.
randomIndices = randperm(size(X, 1));
selectedExamples = randomIndices(1:100);
NN.displayData(X(selectedExamples, :));

fprintf('Press enter to continue.\n');
pause;

%% ================ Part 2: Load pre-trained network parameters ================
fprintf('\nLoading saved neural network parameters...\n');
load('weights2.mat');

nnParams = [Theta1(:); Theta2(:)];

% Evaluate the cost with and without regularization.
lambdaValue = 0;
J = NN.NNcostfun(nnParams, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaValue);

lambdaValue = 1;
J = NN.NNcostfun(nnParams, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaValue);

%% ================ Part 3: Initialize network parameters ================
fprintf('\nInitializing neural network parameters...\n');

initialTheta1 = NN.randInitWeights(inputLayerSize, hiddenLayerSize);
initialTheta2 = NN.randInitWeights(hiddenLayerSize, numLabels);

% Unroll the parameters into a single vector for optimization.
initialNNParams = [initialTheta1(:); initialTheta2(:)];

% Check gradient computation with and without regularization.
NN.checkNNGrad;
lambdaValue = 3;
NN.checkNNGrad(lambdaValue);

debugCost = NN.NNcostfun(nnParams, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaValue);
fprintf('\n\nCost at (fixed) debugging parameters (with lambda = %f): %f\n', ...
    lambdaValue, debugCost);

fprintf('Press enter to continue.\n');
pause;

%% =================== Part 4: Train the neural network ===================
fprintf('\nTraining neural network...\n');

options = optimset('MaxIter', 50);
lambdaValue = 1;

costFunction = @(params) NN.NNcostfun(params, inputLayerSize, hiddenLayerSize, ...
    numLabels, X, y, lambdaValue);

[nnParams, ~] = fmincg(costFunction, initialNNParams, options);

% Reshape the optimized parameter vector back into weight matrices.
Theta1 = reshape(nnParams(1:hiddenLayerSize * (inputLayerSize + 1)), ...
    hiddenLayerSize, inputLayerSize + 1);
Theta2 = reshape(nnParams((1 + hiddenLayerSize * (inputLayerSize + 1)):end), ...
    numLabels, hiddenLayerSize + 1);

fprintf('Press enter to continue.\n');
pause;

%% ================= Part 5: Visualize the learned weights =================
fprintf('\nVisualizing neural network...\n');
NN.displayData(Theta1(:, 2:end));

fprintf('\nPress enter to continue.\n');
pause;

%% ================= Part 6: Make predictions =================
predictions = NN.predict2(Theta1, Theta2, X);
trainingAccuracy = mean(double(predictions == y)) * 100;
fprintf('\nTraining set accuracy: %f\n', trainingAccuracy);

