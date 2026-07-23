%% Support Vector Machines
clear ; close all; clc

%% =============== Part 1: Loading and Visualizing Data ================
fprintf('Loading and Visualizing Data ...\n')
% You will have X, y in your environment
load('data1.mat');
SVM.plot(X, y);
fprintf('Press enter to continue.\n');
pause;

%% ==================== Part 2: Training Linear SVM ====================
% You will have X, y in your environment
load('data1.mat');
fprintf('\nTraining Linear SVM ...\n')
% boundary varies (e.g., try C = 1000)
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
SVM.BoundaryLin(X, y, model);
fprintf('Press enter to continue.\n');
pause;

%% =============== Part 3: Implementing Gaussian Kernel ===============
fprintf('\nEvaluating the Gaussian Kernel ...\n')
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = SVM.gaussianKernel(x1, x2, sigma);
fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :' ...
         '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n'], sigma, sim);
fprintf('Press enter to continue.\n');
pause;

%% =============== Part 4: Visualizing Dataset 2 ================
fprintf('Loading and Visualizing Data ...\n')
% You will have X, y in your environment
load('data2.mat');
SVM.plot(X, y);
fprintf('Press enter to continue.\n');
pause;

%% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');
% You will have X, y in your environment
load('data2.mat');
C = 1; 
sigma = 0.1;
% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
model= svmTrain(X, y, C, @(x1, x2) SVM.gaussianKernel(x1, x2, sigma)); 
SVM.Boundary(X, y, model);
fprintf('Press enter to continue.\n');
pause;

%% =============== Part 6: Visualizing Dataset 3 ================
fprintf('Loading and Visualizing Data ...\n')
% You will have X, y in your environment
load('data3.mat');
SVM.plot(X, y);
fprintf('Press enter to continue.\n');
pause;

%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
load('data3.mat');

% Try different SVM Parameters here
[C, sigma] = SVM.dataset3Params(X, y, Xval, yval);

% Train the SVM
model= svmTrain(X, y, C, @(x1, x2) SVM.gaussianKernel(x1, x2, sigma));
SVM.Boundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

