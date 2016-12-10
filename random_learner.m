function [model, errors, costs] = random_learner(data, labels, buget, ...
    testd, testl, base_learner, report_step, initial_seed)
    n = size(data, 1);
    nt = size(testd, 1);
    active_idxs = 1 : n;
    R = zeros(n, 1);
    errors = [];
    costs = [];
    counter = 0;
    
    while initial_seed > 0
        query_index = random_draw(active_idxs(active_idxs(R == 0)));
        R(query_index) = 1;
        counter = counter + 1;
        initial_seed = initial_seed - 1;
    end
    
    model = base_learner.learn(data(R == 1, :), labels(R == 1), []);
    ytest_predict = base_learner.predict(model, testd);
    errors = [errors; sum(ytest_predict ~= testl) / nt];
    costs = [costs; counter];
    for i = 1 : buget - initial_seed
        query_index = random_draw(active_idxs(active_idxs(R == 0)));
        R(query_index) = 1;
        counter = counter + 1;
        if mod(i, report_step) == 1
            model = base_learner.learn(data(R == 1, :), labels(R == 1), model);
            ytest_predict = base_learner.predict(model, testd);
            errors = [errors; sum(ytest_predict ~= testl) / nt];
            costs = [costs; counter];
            disp(['cost = ', num2str(i), ' error rate = ', num2str(errors(end))]);
        end
    end
end
        
        
        