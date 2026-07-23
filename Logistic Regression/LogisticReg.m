%% Logistic Regression
clear ; close all; clc

data = load('data1.txt');
X = data(:, [1, 2]); y = data(:, 3);
fprintf(['Plotting data with + (y = 1) examples and o (y = 0) examples.\n']);
figure; 
hold on;
pos = find(y==1); neg = find(y == 0);
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;
fprintf('Press enter to continue.\n');
pause;

[m, n] = size(X);
X = [ones(m, 1) X]; % Add intercept term to x and X_test
initial_theta = zeros(n + 1, 1);

%% ============= Part 3: Optimizing using fminunc  =============
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] =	fminunc(@(t)(LogR.costfun(t, X, y)), initial_theta, options);
fprintf('theta: \n');
fprintf(' %f \n', theta);

LogR.boundary(theta, X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;
fprintf('\nPress enter to continue.\n');
pause;

%% ============== Part 4: Predict and Accuracies ==============
prob = sigmoid([1 45 85] * theta);
fprintf(['Probability for scores 45 and 85 is: %f\n'], prob);
p = (sigmoid(X*theta) >= 0.5);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('\n');

