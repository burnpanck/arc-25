def solution(input: Image) -> Image:
    shapes = sorted(
        find_objects(input, exclude=BLACK, connectivity=8),
        key=lambda obj:obj.count(),
        reverse=True,
    )
    output = make_canvas((len(shapes),1))
    for row,shape in enumerate(shapes):
        c, = most_common_colors(apply_mask(input, shape))
        output = stroke(output, [Coord(row,0)], c)
    return output
