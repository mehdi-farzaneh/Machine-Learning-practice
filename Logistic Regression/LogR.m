classdef LogR

    methods (Static)

        function boundary(theta, X, y)
            XX = X(:,2:3);
            figure;
            hold on;
            pos = find(y==1); neg = find(y == 0);
            plot(XX(pos, 1), XX(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
            plot(XX(neg, 1), XX(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
            hold on

            if size(X, 2) <= 3
                % Only need 2 points to define a line, so choose two endpoints
                plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
                % Calculate the decision boundary line
                plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
                % Plot, and adjust axes for better viewing
                plot(plot_x, plot_y)
                legend('Admitted', 'Not admitted', 'Decision Boundary')
                axis([30, 100, 30, 100])
            else
                % Here is the grid range
                u = linspace(-1, 1.5, 50);
                v = linspace(-1, 1.5, 50);
                z = zeros(length(u), length(v));
                % Evaluate z = theta*x over the grid
                for i = 1:length(u)
                    for j = 1:length(v)
                        z(i,j) = LogR.mapping(u(i), v(j))*theta;
                    end
                end
                z = z'; % important to transpose z before calling contour
                contour(u, v, z, [0, 0], 'LineWidth', 2)
            end
            hold off
        end


        function [J, grad] = costfun(theta, X, y)
            m = length(y); % number of training examples

            J = (1/m)* sum((-y'*log(sigmoid(X*theta)) - (1-y')*log(1 - ...
                sigmoid(X*theta))));
            grad = (1/m)*[X'*((sigmoid(X * theta))-y)];
        end

        function [J, grad] = costfunRegul(theta, X, y, lambda)

            m = length(y); % number of training examples
            J = 0;
            grad = zeros(size(theta));
            z = zeros(m);
            J = sum((1/m)*[(-y.*(log(sigmoid(X*theta))))-((1-y).*(log(1 - ...
                (sigmoid(X*theta)))))]);
            J = J + (lambda/(2*m))*sum(theta(2:end).^2);

            grad = (1/m)*[X'*((sigmoid(X * theta))-y)];
            temp = theta;
            temp(1) = 0;
            grad = grad + (lambda/m)*temp;
        end

        function out = mapping(X1, X2)
            degree = 6;
            out = ones(size(X1(:,1)));
            for i = 1:degree
                for j = 0:i
                    out(:, end+1) = (X1.^(i-j)).*(X2.^j);
                end
            end
        end

        function [J, grad] = costfunMcc(theta, X, y, lambda)

            m = length(y); % number of training examples
            J = 0;
            grad = zeros(size(theta));

            J = sum((1/m)*[(-y.*(log(sigmoid(X*theta))))-((1-y).*(log(1 - ...
                (sigmoid(X*theta)))))]);
            J = J + (lambda/(2*m))*sum(theta(2:end).^2);

            grad = (1/m)*[X'*((sigmoid(X * theta))-y)];% unregularized gradient
            temp = theta;
            temp(1) = 0;
            grad = grad + (lambda/m)*temp;
            grad = grad(:);

        end

        function [h, display_array] = displayData(X, example_width)
            % Set example_width automatically if not passed in
            if ~exist('example_width', 'var') || isempty(example_width)
            	example_width = round(sqrt(size(X, 2)));
            end

            colormap(gray);% Gray Image
            [m n] = size(X);% Compute rows, cols
            example_height = (n / example_width);

            % Compute number of items to display
            display_rows = floor(sqrt(m));
            display_cols = ceil(m / display_rows);

            pad = 1;% Between images padding
            % Setup blank display
            display_array = - ones(pad + display_rows * (example_height ...
                + pad), pad + display_cols * (example_width + pad));

            % Copy each example into a patch on the display array
            curr_ex = 1;
            for j = 1:display_rows
            	for i = 1:display_cols
            		if curr_ex > m,
            			break;
            		end
            		% Copy the patch
            		% Get the max value of the patch
            		max_val = max(abs(X(curr_ex, :)));
            		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
  		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
                      reshape(X(curr_ex, :), example_height, example_width) / max_val;
            		curr_ex = curr_ex + 1;
            	end
            	if curr_ex > m,
            		break;
            	end
            end

            h = imagesc(display_array, [-1 1]); % Display Image
            axis image off % Do not show axis
            drawnow;
        end

    end

end