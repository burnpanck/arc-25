def solution(input: Image) -> AnyImage:
    bg_mask = mask_color(input, BLACK)
    # find the "ladder"; the object with the largest bounding box
    ladder = max(
        find_objects(input, exclude=BLACK),
        key = lambda obj:find_bbox(obj).area,
    )
    bbox = find_bbox(ladder)
    rect_mask = rect_to_mask(input, bbox)
    color, = most_common_colors(apply_mask(input, ladder))
    # try drawing a pattern in both axes, selecting
    # the one which does not paint over black cells
    for dir in [RIGHT, DOWN]:
        # fill stripe pattern into candidate output
        candidate = fill(
            input,
            [color, None],
            dir=dir,
            clip=rect_mask,
            pattern_origin=bbox.topleft,
        )
        # stop when black cells remain untouched
        if (mask_color(candidate, BLACK) == bg_mask).all():
            break
    # add frame
    frame_mask = rect_mask&~erode(rect_mask)
    output = fill(candidate, color, clip=frame_mask)
    return output
