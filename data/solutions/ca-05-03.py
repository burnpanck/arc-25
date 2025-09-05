def solution(input: Canvas) -> Canvas:
    n_obj = len(list(find_objects(input, exclude=BLACK)))
    output = make_canvas(1,n_obj)
    output = fill(output, GREEN)
    return output
