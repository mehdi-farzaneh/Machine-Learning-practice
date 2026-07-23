%% Regularized Linear Regression and Bias-Variance
clear ; close all; clc

fprintf('Loading and Visualizing Data ...\n')
load ('ex5data1.mat');% X, y, Xval, yval, Xtest, ytest
m = size(X, 1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
fprintf('Press enter to continue.\n');
pause;

%% =========== Part 5: Learning Curve for Linear Regression =============
lambda = 0;
[error_train, error_val] = LinR.learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, lambda);
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])
fprintf('Press enter to continue.\n');
pause;

%% =========== Part 6: Feature Mapping for Polynomial Regression =========
p = 8;
X_poly = c_polyFeat(X, p);
[X_poly, mu, sigma] = LinR.featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];  % Add Ones

X_poly_test = LinR.polyFeat(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test]; % Add Ones

X_poly_val = LinR.polyFeat(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];  % Add Ones

%% =========== Part 7: Learning Curve for Polynomial Regression ==========
lambda = 0;
[theta] = LinR.trainLinReg(X_poly, y, lambda);
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
c_plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

figure(2);
[error_train, error_val] = ...
    LinR.learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', ...
    lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')
fprintf('Press enter to continue.\n');
pause;

%% =========== Part 8: Validation for Selecting Lambda =============
[lambda_vec, error_train, error_val] = ...
    LinR.validationCurve(X_poly, y, X_poly_val, yval);
close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

