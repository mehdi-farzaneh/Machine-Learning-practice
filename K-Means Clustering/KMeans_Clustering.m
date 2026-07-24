%% K-means clustering
clear; close all; clc

%% ============ Part 1: Find closest centroids and compute initial means =========
load('data1.mat');

% Plot the raw data points.
figure;
scatter(X(:, 1), X(:, 2), 15);

K = 3; % Number of clusters
randomIndices = randperm(size(X, 1));
initialCentroids = X(randomIndices(1:K), :);

% Assign each point to the nearest centroid and recompute cluster centers.
idx = KMC.findCent(X, initialCentroids);
centroids = KMC.computeCent(X, idx, K);

fprintf('\nCentroids computed after initial assignment of points to clusters:\n');
fprintf(' %f %f \n', centroids');

fprintf('\nPress enter to continue.\n');
pause;

%% =================== Part 2: Run k-means clustering ======================
fprintf('\nRunning k-means clustering on the example dataset.\n');
load('data1.mat');

K = 3;
maxIters = 10;
randomIndices = randperm(size(X, 1));
initialCentroids = X(randomIndices(1:K), :);

[finalCentroids, clusterIndices] = KMC.kMeans(X, initialCentroids, maxIters);

figure;
hold on;
KMC.plotResult(X, finalCentroids, initialCentroids, clusterIndices, K);

fprintf('\nCentroids after clustering:\n');
fprintf(' %f %f \n', finalCentroids');

