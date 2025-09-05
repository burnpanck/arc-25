def solution(input: Canvas) -> Canvas:
    h,w = input.shape
    # create canvas twice the size in both dimensions
    output = make_canvas(2*h,2*w)
    # paste the input into the output canvas at the top-left corner
    output = paste(output, input, topleft=(0,0))
    # symmetrically extend across both axes
    for op in [FLIP_LR, FLIP_UD]:
        # make a copy flipped about the current axis
        transformed = transform(output, op)
        # paste the flippped copy over the original
        output = paste(output, transformed)
    return output
