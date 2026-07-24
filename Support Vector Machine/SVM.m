classdef SVM

    methods (Static)

        function BoundaryLin(X, y, model)
            % Plot the decision boundary learned by a linear SVM.
            w = model.w;
            b = model.b;
            xPoints = linspace(min(X(:, 1)), max(X(:, 1)), 100);
            yPoints = -(w(1) * xPoints + b) / w(2);

            SVM.plot(X, y);
            hold on;
            plot(xPoints, yPoints, '-b');
            hold off;
        end

        function Boundary(X, y, model, varargin)
            % Plot the decision boundary learned by a non-linear SVM.
            SVM.plot(X, y);
            x1Plot = linspace(min(X(:, 1)), max(X(:, 1)), 100)';
            x2Plot = linspace(min(X(:, 2)), max(X(:, 2)), 100)';
            [X1, X2] = meshgrid(x1Plot, x2Plot);

            values = zeros(size(X1));
            for i = 1:size(X1, 2)
                samplePoints = [X1(:, i), X2(:, i)];
                values(:, i) = SVM.svmPredict(model, samplePoints);
            end

            hold on;
            contour(X1, X2, values, [0.5 0.5], 'b');
            hold off;
        end

        function [C, sigma] = dataset3Params(X, y, Xval, yval)
            % Select the best SVM hyperparameters using the validation set.
            paramCValues = [0.01 0.03 0.1 0.3 1 3 10 30];
            paramSigmaValues = [0.01 0.03 0.1 0.3 1 3 10 30];
            results = zeros(length(paramCValues) * length(paramSigmaValues), 3);
            rowIndex = 1;

            for paramC = paramCValues
                for paramSigma = paramSigmaValues
                    model = svmTrain(X, y, paramC, @(x1, x2) SVM.gaussianKernel(x1, x2, paramSigma));
                    predictions = SVM.svmPredict(model, Xval);
                    errorRate = mean(double(predictions ~= yval));
                    results(rowIndex, :) = [paramC, paramSigma, errorRate];
                    rowIndex = rowIndex + 1;
                end
            end

            [~, bestIndex] = min(results(:, 3));
            C = results(bestIndex, 1);
            sigma = results(bestIndex, 2);
        end

        function sim = gaussianKernel(x1, x2, sigma)
            % Compute the Gaussian RBF kernel between two vectors.
            x1 = x1(:);
            x2 = x2(:);
            squaredDistance = sum((x1 - x2).^2);
            sim = exp(-(squaredDistance) / (2 * (sigma^2)));
        end

        function sim = linearKernel(x1, x2)
            % Return the linear kernel between two vectors.
            x1 = x1(:);
            x2 = x2(:);
            sim = x1' * x2;
        end

        function plot(X, y)
            % Plot the training examples for binary classification.
            posIndex = find(y == 1);
            negIndex = find(y == 0);

            plot(X(posIndex, 1), X(posIndex, 2), 'k+', 'LineWidth', 1, 'MarkerSize', 7);
            hold on;
            plot(X(negIndex, 1), X(negIndex, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
            hold off;
        end

        function pred = svmPredict(model, X)
            % Predict class labels for the provided feature matrix using the trained SVM model.
            if size(X, 2) == 1
                X = X';
            end

            m = size(X, 1);
            decisionValues = zeros(m, 1);
            pred = zeros(m, 1);

            if strcmp(func2str(model.kernelFunction), 'linearKernel')
                decisionValues = X * model.w + model.b;
            elseif strfind(func2str(model.kernelFunction), 'gaussianKernel')
                X1 = sum(X.^2, 2);
                X2 = sum(model.X.^2, 2)';
                kernelMatrix = bsxfun(@plus, X1, bsxfun(@plus, X2, -2 * X * model.X'));
                kernelMatrix = model.kernelFunction(1, 0) .^ kernelMatrix;
                kernelMatrix = bsxfun(@times, model.y', kernelMatrix);
                kernelMatrix = bsxfun(@times, model.alphas', kernelMatrix);
                decisionValues = sum(kernelMatrix, 2);
            else
                for i = 1:m
                    predictionValue = 0;
                    for j = 1:size(model.X, 1)
                        predictionValue = predictionValue + ...
                            model.alphas(j) * model.y(j) * ...
                            model.kernelFunction(X(i, :)', model.X(j, :)');
                    end
                    decisionValues(i) = predictionValue + model.b;
                end
            end

            % Convert decision values to binary class labels.
            pred(decisionValues >= 0) = 1;
            pred(decisionValues < 0) = 0;
        end

    end

end