def solution(input: Image) -> AnyImage:
    output = input
    for obj in find_objects(input, exclude=BLACK):
        bbox = find_bbox(obj)
        mask = erode(rect_to_mask(input.shape, bbox))
        c = GREEN if bbox.height>bbox.width else ORANGE
        output = fill(output, c, clip=mask)
    return output
