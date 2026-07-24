%% Support vector machines
clear; close all; clc

%% =============== Part 1: Load and visualize dataset 1 ================
fprintf('Loading and visualizing data...\n');
load('data1.mat');
SVM.plot(X, y);

fprintf('Press enter to continue.\n');
pause;

%% ==================== Part 2: Train a linear SVM ====================
load('data1.mat');
fprintf('\nTraining linear SVM...\n');

% The regularization parameter C controls the margin width.
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
SVM.BoundaryLin(X, y, model);

fprintf('Press enter to continue.\n');
pause;

%% =============== Part 3: Evaluate the Gaussian kernel ===============
fprintf('\nEvaluating the Gaussian kernel...\n');

x1 = [1 2 1];
x2 = [0 4 -1];
sigma = 2;
similarity = SVM.gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f:\n' ...
    '\t%f\n(for sigma = 2, this value should be about 0.324652)\n'], ...
    sigma, similarity);

fprintf('Press enter to continue.\n');
pause;

%% =============== Part 4: Load and visualize dataset 2 ================
fprintf('Loading and visualizing data...\n');
load('data2.mat');
SVM.plot(X, y);

fprintf('Press enter to continue.\n');
pause;

%% ========== Part 5: Train an SVM with an RBF kernel (dataset 2) ==========
fprintf('\nTraining SVM with RBF kernel (this may take 1 to 2 minutes)...\n');
load('data2.mat');

C = 1;
sigma = 0.1;

% Use a looser tolerance and fewer passes to speed up training.
model = svmTrain(X, y, C, @(x1, x2) SVM.gaussianKernel(x1, x2, sigma));
SVM.Boundary(X, y, model);

fprintf('Press enter to continue.\n');
pause;

%% =============== Part 6: Load and visualize dataset 3 ================
fprintf('Loading and visualizing data...\n');
load('data3.mat');
SVM.plot(X, y);

fprintf('Press enter to continue.\n');
pause;

%% ========== Part 7: Train an SVM with an RBF kernel (dataset 3) ==========
load('data3.mat');

% Select the best SVM parameters using the validation set.
[C, sigma] = SVM.dataset3Params(X, y, Xval, yval);

% Train the final SVM model.
model = svmTrain(X, y, C, @(x1, x2) SVM.gaussianKernel(x1, x2, sigma));
SVM.Boundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

