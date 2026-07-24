classdef LinR

    methods (Static)

        function [theta, J_history] = gradientdescent(X, y, theta, alpha, num_iters)
            % Gradient descent for linear regression with a single feature.
            % Updates theta over a fixed number of iterations using the
            % learning rate alpha.
            m = numel(y);
            J_history = zeros(num_iters, 1);

            for iter = 1:num_iters
                predictions = X * theta;
                errors = predictions - y;
                thetaUpdate = (alpha / m) * (X' * errors);
                theta = theta - thetaUpdate;
                J_history(iter) = (1 / (2 * m)) * sum((X * theta - y).^2);
            end
        end

        function [theta, J_history] = gradientdescentMulti(X, y, theta, alpha, num_iters)
            % Gradient descent for linear regression with multiple features.
            m = numel(y);
            J_history = zeros(num_iters, 1);

            for iter = 1:num_iters
                predictions = X * theta;
                errors = predictions - y;
                thetaUpdate = (alpha / m) * (X' * errors);
                theta = theta - thetaUpdate;
                J_history(iter) = (1 / (2 * m)) * sum((X * theta - y).^2);
            end
        end

        function [X_norm, mu, sigma] = featureNormalize(X)
            % Normalize each feature to zero mean and unit standard deviation.
            mu = mean(X);
            X_centered = bsxfun(@minus, X, mu);
            sigma = std(X_centered);
            X_norm = bsxfun(@rdivide, X_centered, sigma);
        end

        function [J, grad] = costfunLinPol(X, y, theta, lambda)
            % Cost and gradient for regularized linear regression with polynomial features.
            m = numel(y);
            thetaReg = theta;
            thetaReg(1) = 0;

            J = (1 / (2 * m)) * sum((X * theta - y).^2) + ...
                (lambda / (2 * m)) * sum(thetaReg(2:end).^2);
            grad = (1 / m) * (X' * (X * theta - y)) + (lambda / m) * thetaReg;
            grad = grad(:);
        end

        function [lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval)
            % Evaluate training and validation errors across a range of regularization values.
            lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
            error_train = zeros(length(lambda_vec), 1);
            error_val = zeros(length(lambda_vec), 1);

            for i = 1:length(lambda_vec)
                lambda = lambda_vec(i);
                theta = LinR.trainLinReg(X, y, lambda);
                error_train(i) = LinR.costfunLinPol(X, y, theta, 0);
                error_val(i) = LinR.costfunLinPol(Xval, yval, theta, 0);
            end
        end

        function [theta] = trainLinReg(X, y, lambda)
            % Train a regularized linear regression model using fmincg.
            initial_theta = zeros(size(X, 2), 1);
            costFunction = @(t) LinR.costfunLinPol(X, y, t, lambda);
            options = optimset('MaxIter', 200, 'GradObj', 'on');
            theta = fmincg(costFunction, initial_theta, options);
        end

        function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
            % Compute training and validation errors for progressively larger training sets.
            m = size(X, 1);
            error_train = zeros(m, 1);
            error_val = zeros(m, 1);

            for i = 1:m
                theta = LinR.trainLinReg(X(1:i, :), y(1:i), lambda);
                error_train(i) = LinR.costfunLinPol(X(1:i, :), y(1:i), theta, 0);
                error_val(i) = LinR.costfunLinPol(Xval, yval, theta, 0);
            end
        end

        function [X_out] = polyFeat(X_in, p)
            % Build polynomial features up to degree p.
            X_out = zeros(numel(X_in), p);
            for i = 1:p
                X_out(:, i) = X_in.^i;
            end
        end

        function plotFit(min_x, max_x, mu, sigma, theta, p)
            % Plot the polynomial regression fit over a range of x values.
            hold on;
            xGrid = (min_x - 15:0.05:max_x + 25)';
            X_poly = LinR.polyFeat(xGrid, p);
            X_poly = bsxfun(@minus, X_poly, mu);
            X_poly = bsxfun(@rdivide, X_poly, sigma);
            X_poly = [ones(size(xGrid, 1), 1), X_poly];
            plot(xGrid, X_poly * theta, '--', 'LineWidth', 2);
            hold off;
        end

    end

end