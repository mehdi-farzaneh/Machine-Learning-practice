%% Linear regression with multiple variables
clear; close all; clc

%% ================= Part 1: Load and normalize features =================
fprintf('Loading data...\n\n');
data = load('ex1data2.txt');

featureMatrix = data(:, 1:2);
targetVector = data(:, 3);
m = numel(targetVector);

% Feature normalization helps gradient descent converge faster.
featureMean = mean(featureMatrix);
featureStd = std(featureMatrix);

X_norm = (featureMatrix - featureMean) ./ featureStd;
X_norm = [ones(m, 1), X_norm];

fprintf('Press enter to continue.\n');
pause;

%% ================= Part 2: Gradient descent =================
alpha = 0.01;
numIters = 400;
initialTheta = zeros(size(X_norm, 2), 1);

[theta, J_history] = LinR.gradientdescentMulti(X_norm, targetVector, ...
    initialTheta, alpha, numIters);

figure;
plot(1:numel(J_history), J_history, '-g', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

fprintf('Theta computed from gradient descent:\n');
fprintf(' %f\n', theta);
fprintf('\n');

% Predict the price for a 1650 sq-ft, 3-bedroom house.
houseFeatures = [1650 3];
houseFeaturesNorm = (houseFeatures - featureMean) ./ featureStd;
price = theta(1) + houseFeaturesNorm(1) * theta(2) + ...
    houseFeaturesNorm(2) * theta(3);

fprintf('Predicted price:\n');
fprintf(' %f\n', price);
fprintf('\n');

fprintf('Press enter to continue.\n');
pause;

%% ================= Part 3: Normal equations =================
fprintf('Solving with normal equations...\n');

% Use the original (unnormalized) features for the closed-form solution.
X_raw = [ones(m, 1), featureMatrix];
thetaNormalEq = pinv(X_raw' * X_raw) * X_raw' * targetVector;

fprintf('Theta computed from the normal equations:\n');
fprintf(' %f\n', thetaNormalEq);
fprintf('\n');

priceNormalEq = thetaNormalEq(1) + houseFeatures(1) * thetaNormalEq(2) + ...
    houseFeatures(2) * thetaNormalEq(3);

fprintf('Predicted price using normal equations:\n');
fprintf(' %f\n', priceNormalEq);
fprintf('\n');

