def solution(input: Canvas) -> Canvas:
    # find the lines (anything non-black)
    lines = ~mask_color(input,BLACK)
    # determine their color
    c, = most_common_colors(apply_mask(input,lines))
    # determine bounding box
    bbox = find_bbox(lines)
    # paint the rectangle
    rect_mask = rect_to_mask(input, bbox)
    rect_mask &= ~erode(rect_mask)
    output = fill(input, c, clip=rect_mask)
    # remove the original lines
    output = fill(output, BLACK, clip=lines)
    return output
