%% Regularized logistic regression
clear; close all; clc

% Load the dataset with two features and binary labels.
data = load('data2.txt');
X = data(:, [1, 2]);
y = data(:, 3);

% Plot the two classes.
figure;
hold on;
posIndex = find(y == 1);
negIndex = find(y == 0);

plot(X(posIndex, 1), X(posIndex, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(negIndex, 1), X(negIndex, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
hold on;
xlabel('Microchip test 1');
ylabel('Microchip test 2');
legend('y = 1', 'y = 0');
hold off;

fprintf('Press enter to continue.\n');
pause;

%% ======== Part 1: Fit regularized logistic regression and evaluate accuracy =======
% Map the original features into a higher-dimensional space.
X_mapped = LogR.mapping(X(:, 1), X(:, 2));

initialTheta = zeros(size(X_mapped, 2), 1);
lambdaValue = 1;
options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, ~, ~] = fminunc(@(t)(LogR.costfunRegul(t, X_mapped, y, lambdaValue)), ...
    initialTheta, options);

% Plot the decision boundary.
LogR.boundary(theta, X_mapped, y);
hold on;
title(sprintf('lambda = %g', lambdaValue));
xlabel('Microchip test 1');
ylabel('Microchip test 2');
legend('y = 1', 'y = 0', 'Decision boundary');
hold off;

% Evaluate training accuracy.
predictedLabels = (sigmoid(X_mapped * theta) >= 0.5);
trainingAccuracy = mean(double(predictedLabels == y)) * 100;
fprintf('Train accuracy: %f\n', trainingAccuracy);

