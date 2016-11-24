function [ypredict, estimated_prob] = multiclass_sparse_logistic_predict(X, W)
    X = [ones(size(X, 1), 1), X];
    estimated_prob = compute_prob(X, W);
    [~, ypredict] = max(estimated_prob, [], 2);
end
    
function p = compute_prob(X, W)
    temp = exp(X * W');
    temp_sum = repmat(sum(temp, 2), 1, size(temp, 2));
    p = temp ./ temp_sum;
end