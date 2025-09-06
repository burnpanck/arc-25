def solution(input: Canvas) -> Canvas:
    objects = mask_color(input, RED)
    output = input
    for obj in find_objects(objects, connectivity=8, gap=1):
        bbox = find_bbox(obj)
        mask = rect_to_mask(output.shape, bbox)
        mask = mask & ~objects
        output = fill(output, YELLOW, clip=mask)
    return output
