def solution(input: Image) -> AnyImage:
    fg_mask = ~mask_color(input, BLACK)
    output = input
    for pos in find_cells(fg_mask):
        path = path_ray(pos, RIGHT, shape=output.shape)
        c = input[pos]
        output = stroke(output, path, Pattern([c,GRAY]))
    return output
