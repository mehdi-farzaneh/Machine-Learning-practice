classdef SVM

    methods (Static)

        function BoundaryLin(X, y, model)
            % plots a linear decision boundary learned by the SVM
            w = model.w;
            b = model.b;
            xp = linspace(min(X(:,1)), max(X(:,1)), 100);
            yp = - (w(1)*xp + b)/w(2);
            SVM.plot(X, y);
            hold on;
            plot(xp, yp, '-b');
            hold off
        end

        function Boundary(X, y, model, varargin)
            % plots a non-linear decision boundary learned by the SVM
            SVM.plot(X, y)
            x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
            x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
            [X1, X2] = meshgrid(x1plot, x2plot);

            vals = zeros(size(X1));
            for i = 1:size(X1, 2)
                this_X = [X1(:, i), X2(:, i)];
                vals(:, i) = SVM.svmPredict(model, this_X);
            end
            hold on
            contour(X1, X2, vals, [0.5 0.5], 'b');
            hold off;
        end

        function [C, sigma] = dataset3Params(X, y, Xval, yval)
            paramc = [0.01 0.03 0.1 0.3 1 3 10 30];
            params = [0.01 0.03 0.1 0.3 1 3 10 30];
            results = zeros(length(paramc) * length(params), 3);
            row = 1;
            for paramc = [0.01 0.03 0.1 0.3 1 3 10 30]
                for params = [0.01 0.03 0.1 0.3 1 3 10 30]
                    model= svmTrain(X, y, paramc, @(x1, x2) SVM.gaussianKernel(x1, x2, params));
                    predictions = SVM.svmPredict(model, Xval);
                    err = mean(double(predictions ~= yval)); %compute the prediction error
                    results(row,:) = [paramc params err];
                    row = row + 1;
                end
            end
            [err_min, i] = min(results(:,3));
            C = results(i,1);
            sigma = results(i,2);
        end

        function sim = gaussianKernel(x1, x2, sigma)
            x1 = x1(:); x2 = x2(:);
            dif = sum((x1 - x2).^2);
            sim = exp(-(dif)/(2*(sigma^2)));
        end

        function sim = linearKernel(x1, x2)
            % returns a linear kernel between x1 and x2
            x1 = x1(:); x2 = x2(:);
            sim = x1' * x2;
        end


        function plot(X, y)
            pos = find(y == 1); neg = find(y == 0);
            plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 1, 'MarkerSize', 7)
            hold on;
            plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
            hold off;
        end


        function pred = svmPredict(model, X)
             if (size(X, 2) == 1)
                X = X';
            end

            m = size(X, 1);
            p = zeros(m, 1);
            pred = zeros(m, 1);

            if strcmp(func2str(model.kernelFunction), 'linearKernel')
                p = X * model.w + model.b;
            elseif strfind(func2str(model.kernelFunction), 'gaussianKernel')
                X1 = sum(X.^2, 2);
                X2 = sum(model.X.^2, 2)';
                K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));
                K = model.kernelFunction(1, 0) .^ K;
                K = bsxfun(@times, model.y', K);
                K = bsxfun(@times, model.alphas', K);
                p = sum(K, 2);
            else
                for i = 1:m
                    prediction = 0;
                    for j = 1:size(model.X, 1)
                        prediction = prediction + ...
                            model.alphas(j) * model.y(j) * ...
                            model.kernelFunction(X(i,:)', model.X(j,:)');
                    end
                    p(i) = prediction + model.b;
                end
            end
            % Convert predictions into 0 / 1
            pred(p >= 0) =  1;
            pred(p <  0) =  0;
        end

    end

end