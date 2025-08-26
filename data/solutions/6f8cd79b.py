def solution(input: Canvas) -> Canvas:
    all = mask_all(input)
    outline = ~erode(all)
    output = paint(input, CYAN, clip=outline)
    return output
