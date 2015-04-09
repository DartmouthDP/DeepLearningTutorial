function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

EPSILON = 0.0001;  
thetaPositive = zeros(size(theta));
thetaNegative = zeros(size(theta));

for i = 1:length(numgrad)
    element = zeros(size(theta));
    element(i) = 1; 
    thetaPositive = theta + EPSILON * element; 
    thetaNegative = theta - EPSILON * element;
    numgrad(i) = (J(thetaPositive) - J(thetaNegative))/(2 * EPSILON);
end


%% ---------------------------------------------------------------
end
