classdef NN

    methods (Static)

        function g = sigGradient(z)
            g = zeros(size(z));
            g = sigmoid(z).*(1-sigmoid(z));
        end

        function p = predict(Theta1, Theta2, X)
            m = size(X, 1);
            num_labels = size(Theta2, 1);
            p = zeros(size(X, 1), 1);
            % Theta1 25*401     Theta2 10*26    X  5000*400
            a1 = [ones(m,1) X]; % 5000*401
            z2 = a1*Theta1';  % 5000*25
            a2 = sigmoid(z2);  %5000*25
            z3 = [ones(m,1) a2]*Theta2'; %5000*10
            a3 = sigmoid(z3); % 5000*10
            [v p] = max(a3, [], 2);
        end

        function p = predict2(Theta1, Theta2, X)
            m = size(X, 1);
            num_labels = size(Theta2, 1);
            p = zeros(size(X, 1), 1);
            h1 = sigmoid([ones(m, 1) X] * Theta1');
            h2 = sigmoid([ones(m, 1) h1] * Theta2');
            [dummy, p] = max(h2, [], 2);
        end

        function checkNNGrad(lambda)
            %Creates a small neural network to check the backpropagation gradients
            if ~exist('lambda', 'var') || isempty(lambda)
                lambda = 0;
            end

            input_layer_size = 3;
            hidden_layer_size = 5;
            num_labels = 3;
            m = 5;

            Theta1 = NN.debugInitializeWeights(hidden_layer_size, input_layer_size);
            Theta2 = NN.debugInitializeWeights(num_labels, hidden_layer_size);
            % Reusing debugInitializeWeights to generate X
            X  = NN.debugInitializeWeights(m, input_layer_size - 1);
            y  = 1 + mod(1:m, num_labels)';
            % Unroll parameters
            nn_params = [Theta1(:) ; Theta2(:)];

            % Short hand for cost function
            costFunc = @(p) NN.NNcostfun(p, input_layer_size, hidden_layer_size, ...
                num_labels, X, y, lambda);
            [cost, grad] = costFunc(nn_params);
            numgrad = NN.computeNumericalGradient(costFunc, nn_params);

            % Visually examine the two gradient computations.  
            disp([numgrad grad]);
            diff = norm(numgrad-grad)/norm(numgrad+grad);
            fprintf('%g\n', diff);
        end

        function numgrad = computeNumericalGradient(J, theta)
            numgrad = zeros(size(theta));
            perturb = zeros(size(theta));
            e = 1e-4;
            for p = 1:numel(theta)
                % Set perturbation vector
                perturb(p) = e;
                loss1 = J(theta - perturb);
                loss2 = J(theta + perturb);
                % Compute Numerical Gradient
                numgrad(p) = (loss2 - loss1) / (2*e);
                perturb(p) = 0;
            end
        end

        function W = debugInitializeWeights(fan_out, fan_in)
            W = zeros(fan_out, 1 + fan_in);
            W = reshape(sin(1:numel(W)), size(W)) / 10;
        end

        function [h, display_array] = displayData(X, example_width)
            % Set example_width automatically if not passed in
            if ~exist('example_width', 'var') || isempty(example_width)
            	example_width = round(sqrt(size(X, 2)));
            end
            colormap(gray);% Gray Image

            [m n] = size(X);
            example_height = (n / example_width);
            % Compute number of items to display
            display_rows = floor(sqrt(m));
            display_cols = ceil(m / display_rows);
            % Between images padding
            pad = 1;

            % Setup blank display
            display_array = - ones(pad + display_rows * (example_height + pad), ...
                pad + display_cols * (example_width + pad));

            % Copy each example into a patch on the display array
            curr_ex = 1;
            for j = 1:display_rows
            	for i = 1:display_cols
            		if curr_ex > m
            			break;
            		end

            		% Get the max value of the patch
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
            axis image off% Do not show axis
            drawnow;
        end


        function [J grad] = NNcostfun(nn_params, ...
                input_layer_size, ...
                hidden_layer_size, ...
                num_labels, ...
                X, y, lambda)

            Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                hidden_layer_size, (input_layer_size + 1));
            Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                num_labels, (hidden_layer_size + 1));

            m = size(X, 1);
            J = 0;
            Theta1_grad = zeros(size(Theta1));
            Theta2_grad = zeros(size(Theta2));

            a1 = [ones(m,1) X];
            z2 = a1*Theta1';
            a2 = [ones(m,1) sigmoid(z2)];
            z3 = a2*Theta2';
            a3 = sigmoid(z3);
            id_matrix = eye(num_labels);
            y_matrix = id_matrix(y,:);
            J = (1/m)*sum(sum((-(y==(1:num_labels)).*(log(a3)))-(1-(y==(1:num_labels))...
                ).*(log(1-(a3))))) + (lambda/(2*m))*sum(sum((Theta1(:,2:end).^2))...
                ) + (lambda/(2*m))*sum(sum((Theta2(:,2:end).^2)));
            d3 = a3 - y_matrix;
            d2 = d3 * Theta2(:,2:end).*NN.sigGradient(z2);
            Delta1 = d2' * a1;
            Delta2 = d3' * a2;
            Theta1(:,1) = 0;
            Theta2(:,1) = 0;
            Theta1 = (lambda/m)*Theta1;
            Theta2 = (lambda/m)*Theta2;
            Theta1_grad = (1/m)*Delta1 + Theta1;
            Theta2_grad = (1/m)*Delta2 + Theta2;

            % Unroll gradients
            grad = [Theta1_grad(:) ; Theta2_grad(:)];
        end

        function W = randInitWeights(L_in, L_out)
            W = zeros(L_out, 1 + L_in);
            epsilon = (2.4495)/(sqrt(L_out+L_in));
            W = rand(L_out, L_in + 1) * 2 * epsilon - epsilon;
        end

    end

end