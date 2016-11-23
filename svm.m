function classifier = svm
    classifier.learn = @learn;
    classifier.predict = @my_predict;
end

function model = learn(X, y)
    model = fitcecoc(X, y);
end

function y = my_predict(model, X)
    y = predict(model, X);
end