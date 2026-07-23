%% Linear Regression - Basic
clear ; close all; clc

%% ==================== Part 1: Plotting =======================
data = load('ex1data1.txt');
x = data(:, 1); 
y = data(:, 2);
m = length(y); % number of training examples
plot (x, y, 'rx')
ylabel ('Profit in $10,000s');
xlabel ('Population of City in 10,000s');
fprintf('Press enter to continue.\n');
pause;

%% =================== Part 2: Gradient descent ===================
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500; % gradient descent settings
alpha = 0.01; % gradient descent settings
theta = LinR.gradientdescent(X, y, theta, alpha, iterations);
hold on;
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off 
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n', ...
    predict1*10000);
fprintf('Press enter to continue.\n');
pause;

%% ============= Part 4: Visualizing Cost J(theta_0, theta_1) ============
fprintf('Visualizing J(theta_0, theta_1) ...\n')
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

J_vals = zeros(length(theta0_vals), length(theta1_vals));
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
  	  t = [theta0_vals(i); theta1_vals(j)];
	    J_vals(i,j) = (1/(2*m))*sum((X*t - y).^2);
    end
end
J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');
fprintf('Press enter to continue.\n');
pause;

figure;

% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

