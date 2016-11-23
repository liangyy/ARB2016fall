function [model, errors] = random_learner(data, labels, buget, testd, testl, base_learner, report_step)
    n = size(data, 1);
    nt = size(testd, 1);
    active_idxs = 1 : n;
    R = zeros(n, 1);
    errors = zeros(buget, 1);
    for i = 1 : buget
        query_index = random_draw(active_idxs(active_idxs(R == 0)));
        R(query_index) = 1;
        if mod(i, report_step) == 1
            model = base_learner.learn(data(R == 1, :), labels(R == 1));
            ytest_predict = base_learner.predict(model, testd);
            errors(i) = sum(ytest_predict ~= testl) / nt;
            disp(['cost = ', num2str(i), ' error rate = ', num2str(errors(i))]);
        end
    end
end

function query_index = random_draw(indexs)
    n = size(indexs);
    temp = randi(n);
    query_index = indexs(temp);
end
        
        
        