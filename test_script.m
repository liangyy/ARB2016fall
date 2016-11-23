%% test random learner

% load easy data
filename = 'Data/EASY_TRAIN.csv';
test_filename = 'Data/EASY_TEST.csv';
[easy_x, easy_y, encoding] = read_mydata(filename);
[easy_test_x, easy_test_y, ~] = read_mydata(filename);

% test
buget = 2500;
report_step = 20;
base_learner = svm;
[model, errors] = random_learner(easy_x, easy_y, buget, easy_test_x, easy_test_y, base_learner, report_step);
