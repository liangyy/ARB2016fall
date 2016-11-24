function classifier = learner_sparselogit(lambda1, lambda2, stop)
    classifier.learn = @(x, y, model0) learn(x, y, model0, lambda1, lambda2, stop);
    classifier.predict = @(model, x) my_predict(model, x);
end

function model = learn(x, y, model0, lambda1, lambda2, stop)
    model = multiclass_sparse_logistic_regression(x, y, model0, stop, lambda1, lambda2);
end

function [y, prob_estimates] = my_predict(model, x)
    [y, prob_estimates] = multiclass_sparse_logistic_predict(x, model);
end