def solution(input: Canvas) -> Canvas:
    h,w = input.shape
    # search horizontal separator by looking for a single-line pattern
    best_err = w
    best_row = None
    for row in range(h):
        rect = Rect.make(top=row,left=0,right=int(w)-1,height=1)
        err = pattern_error(apply_mask(input, rect), (1,2))
        if err < best_err:
            best_err = err
            best_row = row
    # fill everything below separator in black
    output = fill(input, BLACK, clip=Rect.make(top=best_row+1,left=0,bottomright=(h,w)))
    return output
