def solution(input: Canvas) -> Canvas:
    fg_mask = ~mask_color(input, BLACK)
    output = input
    never = mask_none(output)
    for pos in find_cells(fg_mask):
        path = path_ray(pos, RIGHT, stop=never)
        c = input[pos]
        output = stroke(output, path, Pattern([c,GRAY]))
    return output
