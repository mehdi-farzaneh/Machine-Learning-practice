classdef LogR

    methods (Static)

        function boundary(theta, X, y)
            % Plot a decision boundary for binary classification.
            featureMatrix = X(:, 2:3);
            figure;
            hold on;

            posIndex = find(y == 1);
            negIndex = find(y == 0);

            plot(featureMatrix(posIndex, 1), featureMatrix(posIndex, 2), 'k+', ...
                'LineWidth', 2, 'MarkerSize', 7);
            plot(featureMatrix(negIndex, 1), featureMatrix(negIndex, 2), 'ko', ...
                'MarkerFaceColor', 'y', 'MarkerSize', 7);

            if size(X, 2) <= 3
                % With a linear decision boundary, use two points to define the line.
                plotX = [min(X(:, 2)) - 2, max(X(:, 2)) + 2];
                plotY = (-1 ./ theta(3)) .* (theta(2) .* plotX + theta(1));
                plot(plotX, plotY);
                legend('Admitted', 'Not admitted', 'Decision Boundary');
                axis([30, 100, 30, 100]);
            else
                % For higher-dimensional mapped features, draw a contour over a grid.
                u = linspace(-1, 1.5, 50);
                v = linspace(-1, 1.5, 50);
                z = zeros(length(u), length(v));

                for i = 1:length(u)
                    for j = 1:length(v)
                        z(i, j) = LogR.mapping(u(i), v(j)) * theta;
                    end
                end

                z = z';
                contour(u, v, z, [0, 0], 'LineWidth', 2);
            end

            hold off;
        end

        function [J, grad] = costfun(theta, X, y)
            % Cost and gradient for logistic regression without regularization.
            m = numel(y);
            predictions = sigmoid(X * theta);

            J = (1 / m) * sum(-y .* log(predictions) - (1 - y) .* log(1 - predictions));
            grad = (1 / m) * (X' * (predictions - y));
        end

        function [J, grad] = costfunRegul(theta, X, y, lambda)
            % Cost and gradient for regularized logistic regression.
            m = numel(y);
            predictions = sigmoid(X * theta);

            J = sum((1 / m) * (-y .* log(predictions) - (1 - y) .* log(1 - predictions)));
            J = J + (lambda / (2 * m)) * sum(theta(2:end).^2);

            grad = (1 / m) * (X' * (predictions - y));
            thetaReg = theta;
            thetaReg(1) = 0;
            grad = grad + (lambda / m) * thetaReg;
        end

        function out = mapping(X1, X2)
            % Create polynomial features of degree 6.
            degree = 6;
            out = ones(size(X1(:, 1)));

            for i = 1:degree
                for j = 0:i
                    out(:, end + 1) = (X1.^(i - j)) .* (X2.^j);
                end
            end
        end

        function [J, grad] = costfunMcc(theta, X, y, lambda)
            % Cost and gradient for multi-class logistic regression using one-vs-all.
            m = numel(y);
            predictions = sigmoid(X * theta);

            J = sum((1 / m) * (-y .* log(predictions) - (1 - y) .* log(1 - predictions)));
            J = J + (lambda / (2 * m)) * sum(theta(2:end).^2);

            grad = (1 / m) * (X' * (predictions - y));
            thetaReg = theta;
            thetaReg(1) = 0;
            grad = grad + (lambda / m) * thetaReg;
            grad = grad(:);
        end

        function [h, display_array] = displayData(X, example_width)
            % Display a set of examples as a tiled grayscale image.
            if ~exist('example_width', 'var') || isempty(example_width)
                example_width = round(sqrt(size(X, 2)));
            end

            colormap(gray);
            [m, n] = size(X);
            example_height = n / example_width;

            display_rows = floor(sqrt(m));
            display_cols = ceil(m / display_rows);

            pad = 1;
            display_array = -ones(pad + display_rows * (example_height + pad), ...
                pad + display_cols * (example_width + pad));

            curr_ex = 1;
            for j = 1:display_rows
                for i = 1:display_cols
                    if curr_ex > m
                        break;
                    end

                    max_val = max(abs(X(curr_ex, :)));
                    display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
                        pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
                        reshape(X(curr_ex, :), example_height, example_width) / max_val;
                    curr_ex = curr_ex + 1;
                end
                if curr_ex > m
                    break;
                end
            end

            h = imagesc(display_array, [-1 1]);
            axis image off;
            drawnow;
        end

    end

end