%% Linear regression - basic
clear; close all; clc

%% ==================== Part 1: Plot the training data =======================
data = load('ex1data1.txt');

featureValues = data(:, 1);
targetValues = data(:, 2);
numExamples = numel(targetValues);

figure;
plot(featureValues, targetValues, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
ylabel('Profit in $10,000s');
xlabel('Population of city in 10,000s');
title('Training data');

fprintf('Press enter to continue.\n');
pause;

%% =================== Part 2: Gradient descent ===================
% Add a column of ones to represent the intercept term.
X = [ones(numExamples, 1), featureValues];

% Initialize the regression parameters.
theta = zeros(2, 1);
numIterations = 1500;
alpha = 0.01;

% Fit the model using gradient descent.
theta = LinR.gradientdescent(X, targetValues, theta, alpha, numIterations);

hold on;
plot(X(:, 2), X * theta, '-');
legend('Training data', 'Linear regression');
hold off;

% Predict the profit for a city with population 35,000.
prediction = [1, 3.5] * theta;
fprintf('For population = 35,000, we predict a profit of %f\n', ...
    prediction * 10000);

fprintf('Press enter to continue.\n');
pause;

%% ============= Part 3: Visualize the cost function J(theta_0, theta_1) ============
fprintf('Visualizing J(theta_0, theta_1)...\n');

% Define a grid of theta values over which to evaluate the cost.
theta0Vals = linspace(-10, 10, 100);
theta1Vals = linspace(-1, 4, 100);

JVals = zeros(length(theta0Vals), length(theta1Vals));

for i = 1:length(theta0Vals)
    for j = 1:length(theta1Vals)
        thetaCandidate = [theta0Vals(i); theta1Vals(j)];
        JVals(i, j) = (1 / (2 * numExamples)) * sum((X * thetaCandidate - targetValues).^2);
    end
end

JVals = JVals';

figure;
surf(theta0Vals, theta1Vals, JVals);
xlabel('\theta_0');
ylabel('\theta_1');
zlabel('Cost J');

fprintf('Press enter to continue.\n');
pause;

figure;

% Plot cost contours to visualize the optimization landscape.
contour(theta0Vals, theta1Vals, JVals, logspace(-2, 3, 20));
xlabel('\theta_0');
ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

