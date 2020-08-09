function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
h_theta = sigmoid(X*theta);
J = (1/m).*(-y'*log(h_theta) - (1-y)'*(log(1-h_theta))) + (lambda/(2*m))*sum(theta(2:length(theta)).^2);

grad_0 = 1/m.*X(:, 1)'*(h_theta-y);
grad_j = 1/m.*X(:, 2:length(theta))'*(h_theta-y) + (lambda/m)*theta(2:length(theta));
grad = [grad_0; grad_j];

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
