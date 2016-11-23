function [data, labels, encoding] = read_mydata(filename)
    temp = readtable(filename, 'Delimiter',',', 'ReadVariableNames', false);
    data = table2array(temp(:, 1:26));
    labels = table2array(temp(:, 27));
    encoding = sort(unique(labels));
    temp_index = 1 : size(encoding, 1);
    labels = cellfun(@(x) temp_index(strcmp(encoding, x) == 1), labels);
end
        
