%% Regularized linear regression and bias-variance
clear; close all; clc

fprintf('Loading and visualizing data...\n');
load('ex5data1.mat'); % Loads X, y, Xval, yval, Xtest, ytest

numTrainExamples = size(X, 1);

% Plot the training data.
figure;
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title('Training data');

fprintf('Press enter to continue.\n');
pause;

%% =========== Part 5: Learning curve for linear regression =============
lambdaValue = 0;

% Build the design matrix for training and validation sets.
X_train = [ones(numTrainExamples, 1), X];
X_val = [ones(size(Xval, 1), 1), Xval];

[errorTrain, errorVal] = LinR.learningCurve(X_train, y, X_val, yval, ...
    lambdaValue);

figure;
plot(1:numTrainExamples, errorTrain, 1:numTrainExamples, errorVal);
title('Learning curve for linear regression');
legend('Train', 'Cross Validation');
xlabel('Number of training examples');
ylabel('Error');
axis([0 13 0 150]);

fprintf('Press enter to continue.\n');
pause;

%% =========== Part 6: Feature mapping for polynomial regression =========
polyDegree = 8;

% Create polynomial features and normalize them.
X_poly = c_polyFeat(X, polyDegree);
[X_poly, mu, sigma] = LinR.featureNormalize(X_poly);
X_poly = [ones(numTrainExamples, 1), X_poly];

X_poly_test = LinR.polyFeat(Xtest, polyDegree);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];

X_poly_val = LinR.polyFeat(Xval, polyDegree);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];

%% =========== Part 7: Learning curve for polynomial regression =========
lambdaValue = 0;
[theta] = LinR.trainLinReg(X_poly, y, lambdaValue);

figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
c_plotFit(min(X), max(X), mu, sigma, theta, polyDegree);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title(sprintf('Polynomial regression fit (lambda = %f)', lambdaValue));

figure(2);
[errorTrain, errorVal] = LinR.learningCurve(X_poly, y, X_poly_val, yval, ...
    lambdaValue);
plot(1:numTrainExamples, errorTrain, 1:numTrainExamples, errorVal);
title(sprintf('Polynomial regression learning curve (lambda = %f)', ...
    lambdaValue));
xlabel('Number of training examples');
ylabel('Error');
axis([0 13 0 100]);
legend('Train', 'Cross Validation');

fprintf('Press enter to continue.\n');
pause;

%% =========== Part 8: Validation for selecting lambda =============
[lambdaVec, errorTrain, errorVal] = LinR.validationCurve(X_poly, y, ...
    X_poly_val, yval);

close all;
figure;
plot(lambdaVec, errorTrain, lambdaVec, errorVal);
legend('Train', 'Cross Validation');
xlabel('\lambda');
ylabel('Error');

