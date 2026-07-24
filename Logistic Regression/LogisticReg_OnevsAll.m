%% One-vs-all logistic regression
% One-vs-all breaks down an N-class problem into N separate binary classifiers.
% Each classifier is trained to distinguish one class from all others.
% During prediction, the class with the highest probability/score is selected.

clear; close all; clc

inputLayerSize = 400; % 20x20 input images of digits
numLabels = 10;       % Labels range from 1 to 10 ("0" corresponds to label 10)

%% =========== Part 1: Load and visualize the data =============
fprintf('Loading and visualizing data...\n');
load('data1.mat'); % Training data stored in arrays X and y

m = size(X, 1);

% Randomly select 100 examples to display.
randIndices = randperm(m);
selectedExamples = X(randIndices(1:100), :);
LogR.displayData(selectedExamples);

fprintf('Press enter to continue.\n');
pause;

%% ============ Part 2: Train one-vs-all logistic regression classifiers =========
fprintf('\nTraining one-vs-all logistic regression...\n');

lambdaValue = 0.1;
numFeatures = size(X, 2);

% Initialize the parameters for all labels.
allTheta = zeros(numLabels, numFeatures + 1);

% Add a bias term to the feature matrix.
X = [ones(m, 1), X];

initialTheta = zeros(numFeatures + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);

for classIndex = 1:numLabels
    % Train a binary classifier for the current class versus all others.
    classLabels = (y == classIndex);
    allTheta(classIndex, :) = fmincg( ...
        @(theta)(LogR.costfunMcc(theta, X, classLabels, lambdaValue)), ...
        initialTheta, options);
end

fprintf('Press enter to continue.\n');
pause;

%% ================ Part 3: Predict labels for the training set ================
predictionProb = sigmoid(X * allTheta');
[~, predictedLabels] = max(predictionProb, [], 2);

trainingAccuracy = mean(double(predictedLabels == y)) * 100;
fprintf('\nTraining set accuracy: %f\n', trainingAccuracy);


