def solution(input: Canvas) -> Canvas:
    h,w = input.shape
    fg_mask = ~mask_color(input, BLACK)
    # search horizontal separator by finding the row
    # with the smallest deviation from a (1,2)-pattern
    best_err = w
    best_row = None
    for row in range(h):
        row_mask = mask_row(input, row)
        # skip all-black lines; these are not a separator
        if not (row_mask & fg_mask).any():
            continue
        # determine deviation from pattern
        masked_row = apply_mask(input, row_mask)
        err = pattern_error(masked_row, (1,2))
        # if deviation is smaller than previous best,
        # update best.
        if err < best_err:
            best_err = err
            best_row = row
    # fill everything below separator in black
    output = fill(input, BLACK, clip=Rect.make(top=best_row+1,left=0,bottomright=Coord(h-1,w-1)))
    return output
