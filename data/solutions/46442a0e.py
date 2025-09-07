def solution(input: Image) -> AnyImage:
    h,w = input.shape
    output = make_canvas((2*h,2*w))
    tile = paste(output, input, topleft=(0,0))
    for _ in range(4):
        output = paste(output, tile)
        tile = transform(tile, ROTATE_LEFT)
    return output
