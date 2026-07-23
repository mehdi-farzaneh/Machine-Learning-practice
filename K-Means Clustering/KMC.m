classdef KMC

    methods (Static)

        function centroids = computeCent(X, idx, K)
            [m n] = size(X);
            centroids = zeros(K, n);
            for i=1:K
                sel = find(idx == i);
                centroids(i,:) = mean(X(sel,:),1);
            end
        end

        function idx = findCent(X, centroids)
            K = size(centroids, 1);
            idx = zeros(size(X,1), 1);
            m = size(X, 1);
            distance = zeros(m, K);
            for i = 1:K
                D = bsxfun(@minus, X, centroids(i,:));
                distance(:,i) = sum(D.^2,2);
            end
            [value, idx] = min(distance, [], 2);
        end

        function [centroids, idx] = kMeans(X, initial_centroids, max_iters)
            [m n] = size(X);
            K = size(initial_centroids, 1);
            centroids = initial_centroids;
            idx = zeros(m, 1);

            for i=1:max_iters
                idx = KMC.findCent(X, centroids);
                centroids = KMC.computeCent(X, idx, K);
            end
        end

        function plotResult(X, centroids, previous, idx, K)
            palette = hsv(K + 1);
            colors = palette(idx, :);
            scatter(X(:,1), X(:,2), 15, colors);
            % Plot the centroids as black x's
            plot(centroids(:,1), centroids(:,2), 'x', ...
                'MarkerEdgeColor','k', ...
                'MarkerSize', 10, 'LineWidth', 3);

            % Plot the history of the centroids with lines
            for j=1:size(centroids,1)
                p1 = centroids(j, :);
                p2 = previous(j, :);
                plot([p1(1) p2(1)], [p1(2) p2(2)]);
            end
        end

    end

end