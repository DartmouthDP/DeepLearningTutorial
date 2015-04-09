function [D] = temp()

dp = [0.1,0.4;0.9,0.7]; 

D = fprime(dp); 

end

%%
% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end


function fpr = fprime(x)
    fpr =  sigmoid(x) .* (1 - sigmoid(x)); 
end