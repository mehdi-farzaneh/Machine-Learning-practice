%% Multi-class classification with neural networks
clear; close all; clc

inputLayerSize = 400;   % 20x20 input images of digits
hiddenLayerSize = 25;   % 25 hidden units
numLabels = 10;         % 10 labels, from 1 to 10 ("0" is labeled as 10)

%% =========== Part 1: Load and visualize the data =============
fprintf('Loading and visualizing data...\n');
load('data1.mat');

m = size(X, 1);

% Randomly select 100 examples to display.
randomIndices = randperm(size(X, 1));
selectedExamples = randomIndices(1:100);
NN.displayData(X(selectedExamples, :));

fprintf('Press enter to continue.\n');
pause;

%% ================ Part 2: Load saved neural network parameters =============
fprintf('\nLoading saved neural network parameters...\n');
load('weights1.mat');

%% ================= Part 3: Make predictions =================
predictions = NN.predict(Theta1, Theta2, X);
trainingAccuracy = mean(double(predictions == y)) * 100;
fprintf('\nTraining set accuracy: %f\n', trainingAccuracy);

fprintf('Press enter to continue.\n');
pause;

% Randomly permute examples and display them with predictions.
randomPermutation = randperm(m);
for i = 1:m
    fprintf('\nDisplaying example image...\n');
    NN.displayData(X(randomPermutation(i), :));

    prediction = NN.predict(Theta1, Theta2, X(randomPermutation(i), :));
    fprintf('\nNeural network prediction: %d (digit %d)\n', prediction, mod(prediction, 10));

    userInput = input('Paused - press enter to continue, q to exit: ', 's');
    if strcmp(userInput, 'q')
        break;
    end
end

