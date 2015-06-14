function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add first column 1
X = [ones(m,1) X];

% Vector Y
ity = eye(num_labels);

% Coss function
for it = 1:m,
    % Caculate h(x)
    hidden = Theta1 * X(it, :)';
    for i = 1 : length(hidden),
        hidden(i) = sigmoid(hidden(i));
    end
    hidden2 = [1; hidden];
    h = (Theta2 * hidden2);
    for i = 1:length(h),
        h(i) = sigmoid(h(i));
    end
    firtLogTerm = log(h);
    secondLogTerm = log(1.-h);
    iy = ity(:, y(it));
    J = J + sum((-iy.*firtLogTerm - (1.-iy).*secondLogTerm));
    
end

J = 1 / m * J;

tTheta1 = Theta1(:, 2:end);
tTheta2 = Theta2(:, 2:end);

tTheta1 = tTheta1.^2;
tTheta2 = tTheta2.^2;

reg = lambda / (2 * m) * (sum(sum(tTheta1)) + sum(sum(tTheta2)));

J = J + reg;


% Backpropagation
for it = 1:m,
    % First, feedforward step
    a1 = X(it, :)';
    z2 = Theta1 * a1;
    a2 = zeros(size(z2));
    for i = 1 : length(z2),
        a2(i) = sigmoid(z2(i));
    end
    
    a2 = [1; a2];
    z3 = (Theta2 * a2);
    a3 = zeros(size(z3));
    for i = 1:length(z3),
        a3(i) = sigmoid(z3(i));
    end
    
    % Second, caculate bigTheta3
    iy = ity(:, y(it));
    bigTheta3 = a3 - iy;
    
    % Third, caculate hidden layer bigTheta2
    dervides = zeros(size(z2));
    for i = 1:length(z2),
        dervides(i) = sigmoid(z2(i)) * (1 - sigmoid(z2(i)));
    end

    % exclude first column
    bigTheta2 = Theta2(:, 2:end)'* bigTheta3 .* dervides;
    
    % Fourth, caculate new thetas
    Theta2_grad = Theta2_grad + bigTheta3 * (a2)';
    Theta1_grad = Theta1_grad + bigTheta2 * (a1)';
  
end

% Fifth, divide by 1 / m
Theta1_grad = (1 / m) .* Theta1_grad;
Theta2_grad = (1 / m) .* Theta2_grad;

% Regulation
Theta1_reg = (lambda / m) .* Theta1(:, 2:end);
Theta2_reg = (lambda / m) .* Theta2(:, 2:end);

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + Theta1_reg;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + Theta2_reg;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
