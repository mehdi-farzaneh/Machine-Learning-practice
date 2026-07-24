classdef NN

    methods (Static)

        function g = sigGradient(z)
            % Derivative of the sigmoid activation function.
            g = sigmoid(z) .* (1 - sigmoid(z));
        end

        function p = predict(Theta1, Theta2, X)
            % Predict class labels for a set of input examples using a two-layer neural network.
            m = size(X, 1);
            a1 = [ones(m, 1), X];
            z2 = a1 * Theta1';
            a2 = sigmoid(z2);
            z3 = [ones(m, 1), a2] * Theta2';
            a3 = sigmoid(z3);
            [~, p] = max(a3, [], 2);
        end

        function p = predict2(Theta1, Theta2, X)
            % Predict class labels using a slightly different forward-pass structure.
            m = size(X, 1);
            hiddenLayerActivations = sigmoid([ones(m, 1), X] * Theta1');
            outputActivations = sigmoid([ones(m, 1), hiddenLayerActivations] * Theta2');
            [~, p] = max(outputActivations, [], 2);
        end

        function checkNNGrad(lambda)
            % Create a small neural network to validate the backpropagation gradients.
            if ~exist('lambda', 'var') || isempty(lambda)
                lambda = 0;
            end

            inputLayerSize = 3;
            hiddenLayerSize = 5;
            numLabels = 3;
            m = 5;

            Theta1 = NN.debugInitializeWeights(hiddenLayerSize, inputLayerSize);
            Theta2 = NN.debugInitializeWeights(numLabels, hiddenLayerSize);
            X = NN.debugInitializeWeights(m, inputLayerSize - 1);
            y = 1 + mod(1:m, numLabels)';

            nnParams = [Theta1(:); Theta2(:)];
            costFunc = @(p) NN.NNcostfun(p, inputLayerSize, hiddenLayerSize, ...
                numLabels, X, y, lambda);
            [cost, grad] = costFunc(nnParams);
            numGrad = NN.computeNumericalGradient(costFunc, nnParams);

            disp([numGrad, grad]);
            diff = norm(numGrad - grad) / norm(numGrad + grad);
            fprintf('%g\n', diff);
        end

        function numgrad = computeNumericalGradient(J, theta)
            % Approximate the gradient numerically for debugging.
            numgrad = zeros(size(theta));
            perturb = zeros(size(theta));
            epsilon = 1e-4;

            for p = 1:numel(theta)
                perturb(p) = epsilon;
                loss1 = J(theta - perturb);
                loss2 = J(theta + perturb);
                numgrad(p) = (loss2 - loss1) / (2 * epsilon);
                perturb(p) = 0;
            end
        end

        function W = debugInitializeWeights(fanOut, fanIn)
            % Initialize a small weight matrix with deterministic values.
            W = zeros(fanOut, 1 + fanIn);
            W = reshape(sin(1:numel(W)), size(W)) / 10;
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

            currExample = 1;
            for j = 1:display_rows
                for i = 1:display_cols
                    if currExample > m
                        break;
                    end

                    max_val = max(abs(X(currExample, :)));
                    display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
                        pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
                        reshape(X(currExample, :), example_height, example_width) / max_val;
                    currExample = currExample + 1;
                end
                if currExample > m
                    break;
                end
            end

            h = imagesc(display_array, [-1 1]);
            axis image off;
            drawnow;
        end

        function [J, grad] = NNcostfun(nnParams, inputLayerSize, hiddenLayerSize, ...
                numLabels, X, y, lambda)
            % Compute the cost and gradients for a two-layer neural network.
            Theta1 = reshape(nnParams(1:hiddenLayerSize * (inputLayerSize + 1)), ...
                hiddenLayerSize, inputLayerSize + 1);
            Theta2 = reshape(nnParams((1 + hiddenLayerSize * (inputLayerSize + 1)):end), ...
                numLabels, hiddenLayerSize + 1);

            m = size(X, 1);
            Theta1Grad = zeros(size(Theta1));
            Theta2Grad = zeros(size(Theta2));

            a1 = [ones(m, 1), X];
            z2 = a1 * Theta1';
            a2 = [ones(m, 1), sigmoid(z2)];
            z3 = a2 * Theta2';
            a3 = sigmoid(z3);

            eyeMatrix = eye(numLabels);
            yMatrix = eyeMatrix(y, :);

            % Compute the cost with regularization.
            J = (1 / m) * sum(sum(-(y == (1:numLabels)) .* log(a3) - ...
                (1 - (y == (1:numLabels))) .* log(1 - a3)))) + ...
                (lambda / (2 * m)) * sum(sum(Theta1(:, 2:end).^2)) + ...
                (lambda / (2 * m)) * sum(sum(Theta2(:, 2:end).^2));

            % Backpropagation gradients.
            delta3 = a3 - yMatrix;
            delta2 = delta3 * Theta2(:, 2:end) .* NN.sigGradient(z2);
            Delta1 = delta2' * a1;
            Delta2 = delta3' * a2;

            % Apply regularization to the weights (excluding bias terms).
            Theta1Reg = Theta1;
            Theta2Reg = Theta2;
            Theta1Reg(:, 1) = 0;
            Theta2Reg(:, 1) = 0;
            Theta1Reg = (lambda / m) * Theta1Reg;
            Theta2Reg = (lambda / m) * Theta2Reg;

            Theta1Grad = (1 / m) * Delta1 + Theta1Reg;
            Theta2Grad = (1 / m) * Delta2 + Theta2Reg;

            grad = [Theta1Grad(:); Theta2Grad(:)];
        end

        function W = randInitWeights(L_in, L_out)
            % Randomly initialize weights for a layer with a small symmetric range.
            W = zeros(L_out, 1 + L_in);
            epsilon = (2.4495) / (sqrt(L_out + L_in));
            W = rand(L_out, L_in + 1) * 2 * epsilon - epsilon;
        end

    end

end