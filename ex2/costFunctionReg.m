function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


lineFunc = X * theta; % N x 1 matrix
sigmoids = sigmoid(lineFunc);
logPosSigmoids = log(sigmoids);
logNegSigmoids = log(1 - sigmoids);

jNormal = 1 / m * sum((-y)' * logPosSigmoids - (1 - y)' * logNegSigmoids); % caculate J
jReg = lambda / (2 * m) * sum(theta(2:length(theta)).^2);
J = jNormal + jReg;

k = length(theta);
grad = 1 / m * (sigmoids - y)' * X;
grad(2:k) = grad(2:k) + lambda / m .* theta(2:k)';



% =============================================================

end
