% objective: nll + lambda1 * |W1|_2^2 + lambda2 * |W1|_2,column + lambda3 * |W2|_2^2
function [W1, W2, objs] = sparse_neuro_net_train(X, y, W01, W02, numhidden, stop_criteria, lambda1, lambda2, lambda3)
    if isempty(W02)
        W02 = rand(8, numhidden + 1); % hard code the number of class here !!
    end
    if isempty(W01)
        W01 = rand(numhidden, size(X, 2) + 1); % hard code the number of class here !!
    end
    y = dummyvar(y);
    X = [ones(size(X, 1), 1), X];
    diff = size(W02, 1) - size(y, 2);
    if diff > 0
        y = [y, zeros(size(y, 1), diff)];
    end
    minstep = 10;
    W1 = W01;
    W2 = W02;
    [~, h1, ~, f] = feed_forward(X, W1, W2);
    objs = compute_nll(f, y) + lambda1 * l2(W1) + lambda2 * l_group(W1) + lambda3 * l2(W2);
    delta = 1;
    counter = 1;
    while delta > stop_criteria || (minstep > counter && delta > 0)
        grad_a2 = - (y - f);
        h1_temp = [ones(size(h1, 1), 1), h1]; 
        grad_W2 = grad_a2' * h1_temp;
        temp = W2;
        temp(:, 1) = [];
        grad_h1 = grad_a2 * temp;
%         ga = g(a1);
        g_prim = h1 .* (1 - h1);
        grad_a1 = grad_h1 .* g_prim;
        grad_W1 = grad_a1' * X;
        
        grad_W2 = grad_W2 + 2 * lambda3 * W2;
        grad_W1 = grad_W1 + 2 * lambda1 * W1;
        
        t = 1;
        beta = 0.5;
        gW = compute_nll(f, y) + lambda1 * l2(W1) + lambda3 * l2(W2);
        while 1
            W1_temp = W1 - t * grad_W1;
            W2_temp = W2 - t * grad_W2;
            Gt1 = (W1 - prox_t_W(W1_temp, lambda2 * t)) / t;   
            W1_temp = W1 - t * Gt1;
            [~, h1, ~, f] = feed_forward(X, W1_temp, W2_temp);
            temp_gW = compute_nll(f, y) + lambda1 * l2(W1_temp) + lambda3 * l2(W2_temp);  
            if temp_gW < gW - t * sum(sum(grad_W1 .* Gt1)) + t / 2 * sum(sum(Gt1 .* Gt1)) - t / 2 * sum(sum(grad_W2 .* grad_W2))
                break
            end
            t = t * beta;
        end
        W1 = W1_temp;
        W2 = W2_temp;
        obj = temp_gW + lambda2 * l_group(W1);
        delta = objs(end) - obj;
        objs = [objs; obj];
%         disp([delta, t]);
        counter = counter + 1;
    end
end

function [a1, h1, a2, f] = feed_forward(X, W1, W2)
    a1 = X * W1';
    h1 = sigmoid_function(a1);
    a2 = [ones(size(h1, 1), 1), h1] * W2';
    temp = exp(a2);
    f = exp(a2) ./ repmat(sum(temp, 2), 1, size(a2, 2));
end

function h1 = sigmoid_function(a1)
    h1 = 1 ./ (1 + exp(-a1));
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