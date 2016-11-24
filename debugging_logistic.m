% debugging sparse logistic regression
filename = 'Data/MODERATE_TRAIN.csv';
test_filename = 'Data/MODERATE_TEST.csv';
[easy_x, easy_y, encoding] = read_mydata(filename);
[easy_test_x, easy_test_y, ~] = read_mydata(test_filename);

lambda1 = 0;
lambda2 = 0;
stop_criteria = 1e-4;
X = easy_x;
y = easy_y;
W0 = zeros(max(y), size(X, 2) + 1);
[W, objs] = multiclass_sparse_logistic_regression(easy_x, easy_y, W0, stop_criteria, lambda1, lambda2);

ypredict = multiclass_sparse_logistic_predict(easy_x, W);
error = sum(ypredict ~= easy_y) / size(easy_y, 1);
disp(['training error = ', num2str(error)]);

ypredict = multiclass_sparse_logistic_predict(easy_test_x, W);
error = sum(ypredict ~= easy_test_y) / size(easy_test_y, 1);
disp(['test error = ', num2str(error)]);


