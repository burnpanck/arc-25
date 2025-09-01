def solution(input: Canvas) -> Canvas:
    output = input
    h,w = input.shape
    # identify foreground cells
    fg_mask = ~mask_color(input, BLACK)
    # paint all columns with more than one foreground cell
    # (there will be just one)
    for col in range(w):
        col_mask = mask_col(input, col)
        mask = col_mask & fg_mask
        if mask.count()>1:
            c, = most_common_colors(apply_mask(input, mask))
            output = fill(output, c, clip=col_mask)
    # same for the row
    for row in range(h):
        row_mask = mask_row(input, row)
        mask = row_mask & fg_mask
        if mask.count()>1:
            c, = most_common_colors(apply_mask(input, mask))
            output = fill(output, c, clip=row_mask)
    return output
