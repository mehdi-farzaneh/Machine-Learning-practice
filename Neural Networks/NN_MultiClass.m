%%  Multi-Class classification with Neural Networks
clear ; close all; clc

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10 ("0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
fprintf('Loading and Visualizing Data ...\n')
load('data1.mat');
m = size(X, 1);
% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
NN.displayData(X(sel, :));
fprintf('Press enter to continue.\n');
pause;

%% ================ Part 2: Loading Pameters ================
fprintf('\nLoading Saved Neural Network Parameters ...\n')
load('weights1.mat');

%% ================= Part 3: Implement Predict =================
pred = NN.predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('Press enter to continue.\n');
pause;

%  Randomly permute examples
rp = randperm(m);
for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    NN.displayData(X(rp(i), :));

    pred = NN.predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end

