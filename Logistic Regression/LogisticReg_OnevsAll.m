%% One-vs-all
% One-vs-all breaks down a N-class problem into N seperate binary classifiers.
% Each classifier is trained to distinguish one class from all other classes.
% In prediction, the class with the highest probability/score is selected.

clear ; close all; clc

input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10 ("0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
fprintf('Loading and Visualizing Data ...\n')
load('data1.mat'); % training data stored in arrays X, y
m = size(X, 1);
rand_indices = randperm(m); % Randomly select 100 data points to display
sel = X(rand_indices(1:100), :);
LogR.displayData(sel);
fprintf('Press enter to continue.\n');
pause;

%% ============ Part 2b: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')
lambda = 0.1;
m = size(X, 1); 
n = size(X, 2); 
all_theta = zeros(num_labels, n + 1); 
X = [ones(m, 1) X]; 
h = zeros(num_labels,1);
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);

for c = 1:num_labels
all_theta(c,:)= fmincg(@(t)(LogR.costfunMcc(t, X, (y == c), lambda)),...
                          initial_theta, options);
end
fprintf('Press enter to continue.\n');
pause;

%% ================ Part 3: Predict for One-Vs-All ================
prediction = sigmoid(X*all_theta');
[v p] = max(prediction, [], 2);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(p == y)) * 100);


