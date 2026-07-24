%% Logistic regression
clear; close all; clc

% Load the dataset containing exam scores and admission labels.
data = load('data1.txt');
X = data(:, [1, 2]);
y = data(:, 3);

fprintf('Plotting data with + (y = 1) examples and o (y = 0) examples.\n');

% Visualize the training data.
figure;
hold on;
posIndex = find(y == 1);
negIndex = find(y == 0);

plot(X(posIndex, 1), X(posIndex, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(negIndex, 1), X(negIndex, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
xlabel('Exam 1 score');
ylabel('Exam 2 score');
legend('Admitted', 'Not admitted');
hold off;

fprintf('Press enter to continue.\n');
pause;

% Add an intercept term for logistic regression.
[m, n] = size(X);
X_with_intercept = [ones(m, 1), X];
initialTheta = zeros(n + 1, 1);

%% ============= Part 3: Optimize using fminunc =============
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, ~] = fminunc(@(t)(LogR.costfun(t, X_with_intercept, y)), initialTheta, options);

fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot the decision boundary.
LogR.boundary(theta, X_with_intercept, y);
hold on;
xlabel('Exam 1 score');
ylabel('Exam 2 score');
legend('Admitted', 'Not admitted');
hold off;

fprintf('\nPress enter to continue.\n');
pause;

%% ============== Part 4: Predict and evaluate accuracy ==============
exampleScores = [1, 45, 85];
probability = sigmoid(exampleScores * theta);
fprintf('Probability for scores 45 and 85 is: %f\n', probability);

predictedLabels = (sigmoid(X_with_intercept * theta) >= 0.5);
trainingAccuracy = mean(double(predictedLabels == y)) * 100;
fprintf('Train accuracy: %f\n', trainingAccuracy);
fprintf('\n');

