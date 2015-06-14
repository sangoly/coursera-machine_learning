function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

lineFunc = X * theta; % N x 1 matrix
sigmoids = sigmoid(lineFunc);
logPosSigmoids = log(sigmoids);
logNegSigmoids = log(1 - sigmoids);

J = 1 / m * sum((-y)' * logPosSigmoids - (1 - y)' * logNegSigmoids); % caculate J
grad = 1 / m * (sigmoids - y)' * X;


% =============================================================

end
