function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

row = size(z, 1);
col = size(z, 2);

for outer = 1 : row,
    for inner = 1 : col,
        g(outer, inner) = 1 / (1 + exp(-z(outer, inner)));
    end
end

% =============================================================

end
