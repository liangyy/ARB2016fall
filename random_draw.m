function query_index = random_draw(indexs)
    n = size(indexs);
    temp = randi(n);
    query_index = indexs(temp);
end