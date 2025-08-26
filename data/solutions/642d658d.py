def solution(input: Canvas) -> Canvas:
    lily_mask = mask_color(input, YELLOW)
    leaf_mask = dilate(lily_mask) & ~lily_mask
    leafs = apply_mask(input, leaf_mask)
    result_color, = most_common_colors(count_colors(leafs))
    output = make_canvas(1,1)
    output = fill(output, result_color)
    return output
