def solution(input: Canvas) -> Canvas:
    # prepare object mask (anything not black)
    fg_mask = ~mask_color(input, BLACK)
    # prepare "marker" mask: object, but not black
    non_blue = fg_mask & ~mask_color(input, BLUE)
    # find the object with the least number of markers
    least_colored = min(
        find_objects(fg_mask),
        key = lambda obj: (obj & non_blue).count(),
    )
    # find its bounding box
    bbox = find_bbox(least_colored)
    # extract that rectangle to the output
    output = extract_image(input, rect=bbox)
    return output
