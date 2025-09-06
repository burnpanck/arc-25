def solution(input: Canvas) -> Canvas:
    lowest = max(
        find_objects(input, exclude=BLACK),
        key = lambda obj: find_bbox(obj).bottom,
    )
    bbox = find_bbox(lowest)
    masked = apply_mask(input, lowest)
    output = make_canvas(bbox.shape, fill=BLACK)
    output = paste(output, masked, topleft=(-bbox.top,-bbox.left))
    return output
