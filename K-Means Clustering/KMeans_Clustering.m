%% K-Means Clustering
clear ; close all; clc

%% ============ Part 1: Find Closest Centroids & Compute Means =========
load('data1.mat');
scatter(X(:,1), X(:,2), 15);
K = 3; % 3 initial Centroids
randidx = randperm(size(X, 1));
initial_centroids = X(randidx(1:K), :);
% Find the closest centroids for the examples
idx = KMC.findCent(X, initial_centroids);
centroids = KMC.computeCent(X, idx, K);

fprintf('\nCentroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');
fprintf('\nPress enter to continue.\n');
pause;

%% =================== Part 2: K-Means Clustering ======================
fprintf('\nRunning K-Means clustering on example dataset.\n');
load('data1.mat');
K = 3;
max_iters = 10;
randidx = randperm(size(X, 1));
initial_centroids = X(randidx(1:K), :);
[centroids2, idx] = KMC.kMeans(X, initial_centroids, max_iters);
figure;
hold on;
KMC.plotResult(X, centroids2, initial_centroids, idx, K);
fprintf('\nCentroids after clustering: \n')
fprintf(' %f %f \n' , centroids2');

