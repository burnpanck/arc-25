def solution(input: Canvas) -> Canvas:
    # find the lowest object (connectivity: 8)
    lowest = max(
        find_objects(input, exclude=BLACK, connectivity=8),
        key = lambda obj: find_bbox(obj).bottom,
    )
    # fill any holes
    lowest = fill_holes(lowest)
    # prepare black output
    output = make_canvas(*input.shape)
    output = fill(output, BLACK)
    # paste lowest object into output
    output = paste(output, apply_mask(input, lowest))
    return output
