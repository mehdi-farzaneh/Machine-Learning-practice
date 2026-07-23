%%  Logistic Regression with Regularization
clear ; close all; clc

data = load('data2.txt');
X = data(:, [1, 2]); y = data(:, 3);
figure; 
hold on;
pos = find(y==1); neg = find(y == 0);
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
hold on;
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')
hold off;
fprintf('Press enter to continue.\n');
pause;

%% ======== Part 1: Regularized Logistic Regression and Accuracies =======
X = LogR.mapping(X(:,1), X(:,2));
initial_theta = zeros(size(X, 2), 1);
lambda = 1;
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exit_flag] = ...
      fminunc(@(t)(LogR.costfunRegul(t, X, y, lambda)), initial_theta, options);

LogR.boundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

p = (sigmoid(X*theta) >= 0.5);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

