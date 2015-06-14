function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% make X have all-one first column
X = [ones(size(X,1),1) X];
secondLayerZ = Theta1 * X';
secondLayerA = zeros(size(secondLayerZ));
for outer=1:size(secondLayerZ, 1),
    for inner=1:size(secondLayerZ, 2),
        secondLayerA(outer, inner) = sigmoid(secondLayerZ(outer, inner));
    end
end

secondLayerA = secondLayerA';
secondLayerA = [ones(size(secondLayerA, 1), 1) secondLayerA];
outputZ = Theta2 * secondLayerA';
for outer=1:size(outputZ, 1),
    for inner=1:size(outputZ, 2),
        outputZ(outer, inner) = sigmoid(outputZ(outer, inner));
    end
end
[Y I] = max(outputZ', [], 2);
p = I(:);




% =========================================================================


end
