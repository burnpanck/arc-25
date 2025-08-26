def solution(input: Canvas) -> Canvas:
    fg_cells = ~mask_color(input, BLACK)
    symmetric = fg_cells & transform(fg_cells, FLIP_LR)
    output = fill(input, BLUE, clip=symmetric)
    return output
