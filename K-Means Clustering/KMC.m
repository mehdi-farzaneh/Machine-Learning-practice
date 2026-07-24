classdef KMC

    methods (Static)

        function centroids = computeCent(X, idx, K)
            % Compute the centroid of each cluster from the assigned points.
            [m, n] = size(X);
            centroids = zeros(K, n);

            for clusterIndex = 1:K
                memberIndices = find(idx == clusterIndex);
                centroids(clusterIndex, :) = mean(X(memberIndices, :), 1);
            end
        end

        function idx = findCent(X, centroids)
            % Assign each point to the nearest centroid.
            K = size(centroids, 1);
            m = size(X, 1);
            idx = zeros(m, 1);
            distances = zeros(m, K);

            for clusterIndex = 1:K
                diff = bsxfun(@minus, X, centroids(clusterIndex, :));
                distances(:, clusterIndex) = sum(diff.^2, 2);
            end

            [~, idx] = min(distances, [], 2);
        end

        function [centroids, idx] = kMeans(X, initialCentroids, maxIters)
            % Run the k-means clustering algorithm for a fixed number of iterations.
            [m, n] = size(X);
            K = size(initialCentroids, 1);
            centroids = initialCentroids;
            idx = zeros(m, 1);

            for iter = 1:maxIters
                idx = KMC.findCent(X, centroids);
                centroids = KMC.computeCent(X, idx, K);
            end
        end

        function plotResult(X, centroids, previousCentroids, idx, K)
            % Plot the clustered data and the movement of centroids over time.
            palette = hsv(K + 1);
            colors = palette(idx, :);

            scatter(X(:, 1), X(:, 2), 15, colors);
            hold on;

            % Plot the final centroids as black x marks.
            plot(centroids(:, 1), centroids(:, 2), 'x', ...
                'MarkerEdgeColor', 'k', ...
                'MarkerSize', 10, 'LineWidth', 3);

            % Plot the path of centroids from the previous step to the current one.
            for clusterIndex = 1:size(centroids, 1)
                currentCentroid = centroids(clusterIndex, :);
                previousCentroid = previousCentroids(clusterIndex, :);
                plot([currentCentroid(1), previousCentroid(1)], ...
                     [currentCentroid(2), previousCentroid(2)]);
            end

            hold off;
        end

    end

end