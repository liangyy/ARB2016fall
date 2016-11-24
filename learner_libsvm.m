function classifier = learner_libsvm(kernel, estimate_prob)
    classifier.learn = @(x, y, model0) learn(x, y, model0, kernel, estimate_prob);
    classifier.predict = @(model, x) my_predict(model, x, estimate_prob);
end

function model = learn(X, y, model0, kernel, estimate_prob)
    cmd = ['-s ', num2str(0), ' -t ', num2str(kernel), ' -b ', num2str(estimate_prob), ' -q 1'];
    model = svmtrain(y, X, cmd);
end

function [y, prob_estimates] = my_predict(model, X, estimate_prob)
    n = size(X, 1);
    cmd = ['-b ', num2str(estimate_prob), ' -q 1'];
    [y, ~, prob_estimates] = svmpredict(zeros(n, 1), X, model, cmd);
end
