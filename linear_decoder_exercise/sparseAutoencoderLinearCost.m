function [cost,grad] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.


% -------------------- YOUR CODE HERE --------------------                                    

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

%% Following are notes on notations by Eric Ding. 

% data: 64x10000 matrix. "data" enters the sparse autoencoder at the input
% layer or layer l1 and serves as labeled response at the output layer. 

% a1: the output of layer l1 is simply the original data.  

% W1: weight parameter matrix connecting layer l1 and l2. The matrix is 25x64 (hiddenSize x visibleSize)
% b1: bias parameter vector connecting layer l1 and l2. The vector is
% 25x1 (hiddenSize x 1)
% z2: model input into layer l2 over all of training data. Matrix size is
% 25x10000
% a2: model output from layer l2 over all of training data. Matrix size is
% 25x10000
% delta2: error matrix at layer l2. Matrix size is 25x10000

% W2: weight parameter matrix connecting layer l2 and l3. The matrix is 64x25 (visibleSize x hiddenSize)
% b2: bias parameter vector connecting layer l2 and l3. The vector is
% 64x1 (visibleSize x 1)
% z3: model input into layer l3 over all of training data. Matrix size is
% 64x10000
% a3: model output from layer l3 over all of training data. Matrix size is
% 64x10000
% delta3: error matrix at layer l3. Matrix size is 64x10000

% W1grad, W2grad, b1grad, b2grad are of the same size as W1, W2, b1, b2. 

% rhoHat2: activation over all of training data at layer l2. It is a 25x1
% vector

%% ------------This is the vectorized version----------
Jcost = 0; %直接误差
Jweight = 0; %权值惩罚
Jsparse = 0; %稀疏性惩罚

[n,m] = size(data);

a1 = data; 

% compute l2 over training data
z2 = W1 * data + repmat(b1,1,m); 
a2 = sigmoid(z2);

% compute l3 over training data
z3 = W2 * a2 + repmat(b2,1,m); 
a3 = z3; 


% 计算预测产生的误差
Jcost = (0.5/m)*sum(sum((a3-data).^2));
% 计算权值惩罚项
Jweight = (1/2)*(sum(sum(W1.^2))+sum(sum(W2.^2)));
% 计算稀释性规则项
rho = (1/m).*sum(a2,2);%求出第一个隐含层的平均值向量
Jsparse = sum(sparsityParam.*log(sparsityParam./rho)+ ...
        (1-sparsityParam).*log((1-sparsityParam)./(1-rho)));
%损失函数的总表达式
cost = Jcost+lambda*Jweight+beta*Jsparse;


% compute delta
delta3 = -(data - a3); 

sparsity_delta = beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho));
%因为加入了稀疏规则项，所以计算偏导时需要引入该项

delta2 = ((W2' * delta3) + repmat(sparsity_delta,1,m)).* fprime(a2); 

% compute gradient 
W1grad = W1grad+delta2*a1';
W1grad = (1/m)*W1grad+lambda*W1;
W2grad = W2grad+delta3*a2';
W2grad = (1/m).*W2grad+lambda*W2;
b1grad = b1grad+sum(delta2,2);
b1grad = (1/m)*b1grad;
b2grad = b2grad+sum(delta3,2);
b2grad = (1/m)*b2grad;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function fpr = fprime(a)

    fpr = a.*(1-a); 
end

