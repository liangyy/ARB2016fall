%% test random learner

% load easy data
% filename = 'Data/EASY_TRAIN.csv';
% test_filename = 'Data/EASY_TEST.csv';
filename = 'Data/MODERATE_TRAIN.csv';
test_filename = 'Data/MODERATE_TEST.csv';
% filename = 'Data/DIFFICULT_TRAIN.csv';
% test_filename = 'Data/DIFFICULT_TEST.csv';

[easy_x, easy_y, encoding] = read_mydata(filename);
[easy_test_x, easy_test_y, ~] = read_mydata(filename);

%% random learner
buget = 450;
report_step = 20;
initial_seed = 10; % initialize with 20 samples randomly selected from the pool
base_learner_random = learner_libsvm(2, 0);
[model_random, errors_random, costs_random] = random_learner(easy_x, easy_y, buget, ...
    easy_test_x, easy_test_y, base_learner_random, report_step, initial_seed);

%% uncertainty learner 
base_learner_uncertainty = learner_libsvm(2, 1);
[model_uncertainty, errors_uncertainty, costs_uncertainty] = uncertainty_based_learner(easy_x, easy_y, buget, ...
    easy_test_x, easy_test_y, base_learner_uncertainty, report_step, initial_seed);

%% plot results
plot(costs_random, errors_random);
hold on;
plot(costs_uncertainty, errors_uncertainty);

