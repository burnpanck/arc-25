def solution(input: Canvas) -> Canvas:
    # prepare output
    m,n = input.shape
    output = make_canvas(3*m,3*n)
    for tile_row in range(3):
        tile = input
        # flip tile for middle row (row 1)
        if tile_row == 1:
            tile = transform(tile, FLIP_LR)
        for tile_col in range(3):
            output = paste(output, tile, at=(m*tile_row, n*tile_col))
    return output
