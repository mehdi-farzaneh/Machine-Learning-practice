classdef LinR

    methods (Static)

        function [theta, J_history] = gradientdescent(X, y, theta, alpha, num_iters)
            %updates theta by taking num_iters gradient steps with learning rate alpha
            m = length(y); % number of training examples
            J_history = zeros(num_iters, 1);
            
            for iter = 1:num_iters
                h = X*theta;
                err = h - y;
                theta_change = (alpha/m)*(X'*err);
                theta = theta - theta_change;
                J_history(iter) = (1/(2*m))*sum((X*theta - y).^2);
            end
        end

        
         function [theta, J_history] = gradientdescentMulti(X, y, theta, alpha, num_iters)
            m = length(y); % number of training examples
            J_history = zeros(num_iters, 1);

            for iter = 1:num_iters
                h = X*theta;
                err = h - y;
                theta_change = (alpha/m)*(X'*err);
                theta = theta - theta_change;
                J_history(iter) = (1/(2*m))*sum((X*theta - y).^2);
            end
         end

         function [X_norm, mu, sigma] = featureNormalize(X)
            mu = mean(X);
            X_norm = bsxfun(@minus, X, mu);
            sigma = std(X_norm);
            X_norm = bsxfun(@rdivide, X_norm, sigma);
         end

         function [J, grad] = costfunLinPol(X, y, theta, lambda)
            m = length(y);
            temp = theta;
            temp(1) = 0;
            J = (1/(2*m))*sum((X*theta - y).^2) + (lambda/(2*m))*sum(theta(2:end).^2);
            grad = (1/m)*(X'*(X*theta - y)) + (lambda/m)*temp;
            grad = grad(:);
         end

         function [lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval)
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
            initial_theta = zeros(size(X, 2), 1);
            costFunction = @(t) LinR.costfunLinPol(X, y, t, lambda);
            options = optimset('MaxIter', 200, 'GradObj', 'on');
            theta = fmincg(costFunction, initial_theta, options);
        end

        function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
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
            X_out = zeros(numel(X_in), p);
            for i = 1:p
                X_out(:, i) = X_in.^i;
            end
        end

        function plotFit(min_x, max_x, mu, sigma, theta, p)
            hold on;
            x = (min_x - 15:0.05:max_x + 25)';
            X_poly = LinR.polyFeat(x, p);
            X_poly = bsxfun(@minus, X_poly, mu);
            X_poly = bsxfun(@rdivide, X_poly, sigma);
            X_poly = [ones(size(x, 1), 1) X_poly];
            plot(x, X_poly * theta, '--', 'LineWidth', 2);
            hold off;
        end

    end
    
end