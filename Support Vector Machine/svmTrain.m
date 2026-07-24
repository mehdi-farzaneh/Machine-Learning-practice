function [model] = svmTrain(X, Y, C, kernelFunction, tol, max_passes)
% Train a support vector machine using a simplified SMO-style algorithm.
% The implementation follows the standard SVM training approach for binary
% classification with a user-supplied kernel function.

if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-3;
end
if ~exist('max_passes', 'var') || isempty(max_passes)
    max_passes = 5;
end

m = size(X, 1);
n = size(X, 2);
Y(Y == 0) = -1;

alphas = zeros(m, 1);
b = 0;
E = zeros(m, 1);
passes = 0;
eta = 0;
L = 0;
H = 0;

% Pre-compute the kernel matrix.
if strcmp(func2str(kernelFunction), 'linearKernel')
    K = X * X';
elseif strfind(func2str(kernelFunction), 'gaussianKernel')
    X2 = sum(X.^2, 2);
    K = bsxfun(@plus, X2, bsxfun(@plus, X2', -2 * (X * X')));
    K = kernelFunction(1, 0) .^ K;
else
    % For custom kernels, compute the matrix directly.
    K = zeros(m);
    for i = 1:m
        for j = i:m
            K(i, j) = kernelFunction(X(i, :)', X(j, :)');
            K(j, i) = K(i, j);
        end
    end
end

fprintf('\nTraining ...');
dots = 12;

while passes < max_passes
    numChangedAlphas = 0;

    for i = 1:m
        % Evaluate the decision function error for the current example.
        E(i) = b + sum(alphas .* Y .* K(:, i)) - Y(i);

        if ((Y(i) * E(i) < -tol && alphas(i) < C) || (Y(i) * E(i) > tol && alphas(i) > 0))
            % Select a second example j at random.
            j = ceil(m * rand());
            while j == i
                j = ceil(m * rand());
            end

            E(j) = b + sum(alphas .* Y .* K(:, j)) - Y(j);

            alphaIOld = alphas(i);
            alphaJOld = alphas(j);

            % Compute bounds L and H for the alpha update.
            if Y(i) == Y(j)
                L = max(0, alphas(j) + alphas(i) - C);
                H = min(C, alphas(j) + alphas(i));
            else
                L = max(0, alphas(j) - alphas(i));
                H = min(C, C + alphas(j) - alphas(i));
            end

            if L == H
                continue;
            end

            % Compute eta for the update step.
            eta = 2 * K(i, j) - K(i, i) - K(j, j);
            if eta >= 0
                continue;
            end

            % Update alpha_j and clip it to the valid range.
            alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;
            alphas(j) = min(H, alphas(j));
            alphas(j) = max(L, alphas(j));

            % If the alpha change is too small, skip the update.
            if abs(alphas(j) - alphaJOld) < tol
                alphas(j) = alphaJOld;
                continue;
            end

            % Update alpha_i.
            alphas(i) = alphas(i) + Y(i) * Y(j) * (alphaJOld - alphas(j));

            % Compute the new bias term using the updated alphas.
            b1 = b - E(i) - Y(i) * (alphas(i) - alphaIOld) * K(i, j)' - ...
                Y(j) * (alphas(j) - alphaJOld) * K(i, j)';
            b2 = b - E(j) - Y(i) * (alphas(i) - alphaIOld) * K(i, j)' - ...
                Y(j) * (alphas(j) - alphaJOld) * K(j, j)';

            if 0 < alphas(i) && alphas(i) < C
                b = b1;
            elseif 0 < alphas(j) && alphas(j) < C
                b = b2;
            else
                b = (b1 + b2) / 2;
            end

            numChangedAlphas = numChangedAlphas + 1;
        end
    end

    if numChangedAlphas == 0
        passes = passes + 1;
    else
        passes = 0;
    end

    fprintf('.');
    dots = dots + 1;
    if dots > 78
        dots = 0;
        fprintf('\n');
    end
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
end
fprintf(' Done! \n\n');

% Save the learned model.
activeIndices = alphas > 0;
model.X = X(activeIndices, :);
model.y = Y(activeIndices);
model.kernelFunction = kernelFunction;
model.b = b;
model.alphas = alphas(activeIndices);
model.w = ((alphas .* Y)' * X)';
end
