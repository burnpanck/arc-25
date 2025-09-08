def solution(input: Image) -> Image:
    rects = [
        find_bbox(obj) for obj in
        find_objects(input, exclude=BLACK)
    ]
    mask = mask_color(input, BLACK)
    output = input
    for axis in range(2):
        oaxis = 1-axis
        a, b = sorted([rect.center[axis] for rect in rects])
        c = 1+max([rect.topleft[oaxis] for rect in rects])
        d = -1+min([rect.bottomright[oaxis] for rect in rects])
        if c>d:
            continue
        if not axis:
            crect = Rect.make(
                top=int(a),
                bottom=int(b),
                left=c,
                right=d,
            )
        else:
            crect = Rect.make(
                left=int(a),
                right=int(b),
                top=c,
                bottom=d,
            )
        cmask = mask & rect_to_mask(output.shape, crect)
        output = fill(output, CYAN, clip=cmask)
    return output
