% adapted from http://stackoverflow.com/questions/6681798/print-variable-name-in-matlab/6682015#6682015
% to retrive variable name as string
function name = varname(v)
    name = inputname(1);
    name = strrep(name, '_', ' ');
end