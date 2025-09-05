def solution(input: Canvas) -> Canvas:
    lengths = sorted(
        obj.count() for obj in
        find_objects(input, exclude=BLACK)
    )
    c, = most_common_colors(input, exclude=BLACK)
    h,w = input.shape
    output = make_canvas(h,w)
    output = fill(output, BLACK)
    for col, length in enumerate(lengths):
        path = path_ray(Coord(h-1,col),UP,stop=mask_none(input))
        output = stroke(output, path[:length], c)
    return output
