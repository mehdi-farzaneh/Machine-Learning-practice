%% Linear regression with multiple variables
clear ; close all; clc

%% ================ Part 1: Feature Normalization ================
fprintf('Load Data...\n\n');
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
% Scale features and set them to zero mean
mu = mean(X);
sigma = std(X);
X = (X - mu)./sigma;
X = [ones(m, 1) X];% Add intercept term to X
fprintf('Press enter to continue.\n');
pause;

%% ================ Part 2: Gradient Descent ================
alpha = 0.01;
num_iters = 400;
theta = zeros(size(X,2), 1);% Init Theta to run Gradient Descent 
[theta, J_history] = LinR.gradientdescentMulti(X, y, theta, alpha, ...
    num_iters);
figure;
plot(1:numel(J_history), J_history, '-g', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');
xx = [1650 3]; % Estimate the price of a 1650 sq-ft, 3 br house
price = theta(1) + (xx(1) - mu(1))*theta(2)/sigma(1) + (xx(2) - ...
    mu(2))*theta(3)/sigma(2); 
fprintf('predicte price: \n');
fprintf(' %f \n', price);
fprintf('\n');
fprintf('Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================
fprintf('Solving with normal equations...\n');
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
X = [ones(m, 1) X];
theta = pinv(X'*X)*X'*y;    %%theta = inv(X'*X)*X'*y;
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');
xx = [1650 3]; % Estimate the price of a 1650 sq-ft, 3 br house
price = theta(1) + xx(1)*theta(2) + xx(2)*theta(3); 

fprintf('new predicte price: \n');
fprintf(' %f \n', price);
fprintf('\n');

