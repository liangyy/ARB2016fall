% reference: http://vision.lbl.gov/Conferences/cvpr/Papers/data/papers/1036.pdf
% title: Multi-Class Active Learning for Image Classification
% main idea: measure the uncertainty using the difference between p_max and p_2nd_max

function [model, errors, costs] = uncertainty_based_learner(data, labels, buget, ...
    testd, testl, base_learner, report_step, initial_seed)
    n = size(data, 1);
    nt = size(testd, 1);
    active_idxs = 1 : n;
    R = zeros(n, 1);
    errors = [];
    costs = [];
    
    % initialize by drawing random samples from the pool
    counter = 0;
    while initial_seed > 0
        query_index = random_draw(active_idxs(active_idxs(R == 0)));
        R(query_index) = 1;
        counter = counter + 1;
        initial_seed = initial_seed - 1;
    end
    model = base_learner.learn(data(R == 1, :), labels(R == 1));
    
    % main body
    for i = 1 : buget - initial_seed
        % compute uncertainty
        [~, estimated_prob] = base_learner.predict(model, data(R ~= 1, :));
        query_index = bvsb(estimated_prob, R);
        R(query_index) = 1;
        counter = counter + 1;
        model = base_learner.learn(data(R == 1, :), labels(R == 1));
        if mod(i, report_step) == 1
            ytest_predict = base_learner.predict(model, testd);
            errors = [errors; sum(ytest_predict ~= testl) / nt];
            costs = [costs; counter];
            disp(['cost = ', num2str(i), ' error rate = ', num2str(errors(end))]);
        end
    end
end

function index = bvsb(estimated_prob, R)
    n = size(R, 1);
    [temp_b, i] = max(estimated_prob, [], 2);
    estimated_prob(:, i) = 0;
    temp_sb = max(estimated_prob, [], 2);
    temp = temp_b - temp_sb;
    [~, i] = min(temp);
    index = 1 : n;
    index = index(R ~= 1);
    index = index(i);
end