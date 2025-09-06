def solution(input: Canvas) -> Canvas:
    open_obj = max(
        find_objects(input, exclude=BLACK, connectivity=8),
        key = lambda obj:fill_holes(obj).count() - obj.count()
    )
    output = make_canvas((1,1))
    c, = most_common_colors(apply_mask(input, open_obj))
    output = fill(output, c)
    return output
