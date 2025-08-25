def solution(input: Canvas) -> Canvas:
    # prepare output
    output = make_canvas(6,6)
    for tile_row in range(3):
        tile = input
        # flip tile for middle row (row 1)
        if tile_row == 1:
            tile = transform(tile, FLIP_LR)
        for tile_col in range(3):
            output = paste(output, tile, at=(2*tile_row, 2*tile_col))
    return output
