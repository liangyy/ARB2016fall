function classifier = learner_sparseNN(lambda1, lambda2, lambda3, numhidden, stop)
    classifier.learn = @(x, y, model0) learn(x, y, model0, lambda1, lambda2, lambda3, numhidden, stop);
    classifier.predict = @(model, x) my_predict(model, x);
end

function model = learn(x, y, model0, lambda1, lambda2,lambda3, numhidden, stop)
    if isempty(model0)
        W01 = [];
        W02 = [];
    else
        W01 = model0{1};
        W02 = model0{2};
    end
    [W1, W2, ~] = sparse_neuro_net_train(x, y, W01, W02, numhidden, stop, lambda1, lambda2, lambda3);
    model{1} = W1;
    model{2} = W2;
end

function [y, prob_estimates] = my_predict(model, x)
    W1 = model{1};
    W2 = model{2};
    [y, prob_estimates] = sparse_neuro_net_predict(x, W1, W2);
end