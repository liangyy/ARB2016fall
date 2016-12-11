%% test random learner
% addpath('libsvm-3.21/matlab');

filename = 'Data/EASY_TRAIN.csv';
test_filename = 'Data/EASY_TEST.csv';
ncol = 26;

[easy_x, easy_y, encoding] = read_mydata(filename, ncol);
[easy_test_x, easy_test_y, ~] = read_mydata(test_filename, ncol);


buget = 2500;
report_step = 20;
%% random learner + sparseNN
rng(0);
initial_seed = 10; % initialize with 20 samples randomly selected from the pool
numhidden = floor(size(easy_x, 2) * 3);
base_learner_sparseNN = learner_sparseNN(1, 1, 1, numhidden, 1e-1);
[model_random, errors_random, costs_random] = random_learner(easy_x, easy_y, buget, ...
    easy_test_x, easy_test_y, base_learner_sparseNN, report_step, initial_seed);

%% uncertainty learner + sparseNN
rng(0);
base_learner_sparseNN = learner_sparseNN(1, 1, 1, numhidden, 1e-1);
[model_uncertainty, errors_uncertainty, costs_uncertainty] = uncertainty_based_learner(easy_x, easy_y, buget, ...
    easy_test_x, easy_test_y, base_learner_sparseNN, report_step, initial_seed);
%% plot results
figure;
[~, trainname] = fileparts(filename);
trainname = strrep(trainname, '_', ' ');
[~, testname] = fileparts(test_filename);
testname = strrep(testname, '_', ' ');
plot(costs_random, errors_random);
hold on;
plot(costs_uncertainty, errors_uncertainty);
h = legend('random', 'uncertainty');
set(h, 'FontSize', 14);
text(0.4, 0.5, ['base learner = ', varname(base_learner_sparseNN)], 'FontSize', 13, 'Units','normalized');
text(0.4, 0.45, ['train data = ', trainname], 'FontSize', 13, 'Units','normalized');
text(0.4, 0.4, ['test data = ', testname], 'FontSize', 13, 'Units','normalized');
saveas(h, 'sparseNN_simple.png', 'png');
%% predict blinded

predict_filename = 'Data/EASY_BLINDED.csv';
easy_predict = csvread(predict_filename);
id = easy_predict(:, 1);
easy_predict_x = easy_predict(:, 2 : size(easy_predict, 2));

[yt, estimated_prob] = base_learner_sparseNN.predict(model_uncertainty, easy_predict_x);
fid = fopen('sparseNN_simple.csv', 'wt');
t = encoding(yt);
for i = 1 : size(id, 1)
    fprintf(fid, '%d,%s\n', id(i), t{i});
end
fclose(fid);