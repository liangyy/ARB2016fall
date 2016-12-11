function [ypredict, estimated_prob] = sparse_neuro_net_predict(X, W1, W2)
    X = [ones(size(X, 1), 1), X];
    [~, ~, ~, estimated_prob] = feed_forward(X, W1, W2);
    [~, ypredict] = max(estimated_prob, [], 2);
end

function [a1, h1, a2, f] = feed_forward(X, W1, W2)
    a1 = X * W1';
    h1 = sigmoid_function(a1);
    a2 = [ones(size(h1, 1), 1), h1] * W2';
    temp = exp(a2);
    f = temp ./ repmat(sum(temp, 2), 1, size(a2, 2));
end

function h1 = sigmoid_function(a1)
    h1 = 1 ./ (1 + exp(-a1));
end
