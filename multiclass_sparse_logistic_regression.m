% objective: nll + lambda1 * |W|_2^2 + lambda2 * |W|_2,column
function [W, objs] = multiclass_sparse_logistic_regression(X, y, W0, stop_criteria, lambda1, lambda2)
    if isempty(W0)
        W0 = zeros(8, size(X, 2) + 1); % hard code the number of class here !!
    end
    y = dummyvar(y);
    X = [ones(size(X, 1), 1), X];
    diff = size(W0, 1) - size(y, 2);
    if diff > 0
        y = [y, zeros(size(y, 1), diff)];
    end
    
    W = W0;
    P = compute_prob(X, W);
    objs = compute_nll(P, y) + lambda1 * l2(W) + lambda2 * l_group(W);
    delta = 1;
    while delta > stop_criteria
        gradW = (P - y)' * X + 2 * lambda1 * W;
        t = 1;
        beta = 0.7;
        gW = compute_nll(P, y) + lambda1 * l2(W);
        while 1
            W_temp = W - t * gradW;
            Gt = (W - prox_t_W(W_temp, lambda2 * t)) / t;   
            W_temp = W - t * Gt;
            P_temp = compute_prob(X, W_temp);
            temp_gW = compute_nll(P_temp, y) + lambda1 * l2(W_temp);
            if temp_gW < gW - t * sum(sum(gradW .* Gt)) + t / 2 * sum(sum(Gt .* Gt))
                break
            end
            t = t * beta;
        end
        W = W_temp;
        P = P_temp;
        obj = temp_gW + lambda2 * l_group(W);
        delta = objs(end) - obj;
        objs = [objs; obj];
%         disp([delta, t]);
    end
end

function p = compute_prob(X, W)
    temp = exp(X * W');
    temp_sum = repmat(sum(temp, 2), 1, size(temp, 2));
    p = temp ./ temp_sum;
end

function nll = compute_nll(P, y)
    temp = sum(P .* y, 2);
    nll = sum(-log(temp));
end

function o = l2(W)
    o = sum(sum(W .^ 2));
end

function o = l_group(W)
    W(:, 1) = [];
    o = sum(sqrt(sum(W .^ 2, 1)));
end

function Wo = prox_t_W(W, l)
    W0 = W(:, 1);
    W(:, 1) = [];
    temp = sqrt(sum(W .^ 2, 1));
    shrink = temp > l;
    out = zeros(size(W));
    coeff = (temp - l) ./ temp;
    coeff = repmat(coeff, size(W, 1), 1);
    out(:, shrink) = coeff(:, shrink) .* W(:, shrink);
    Wo = [W0, out];
end
