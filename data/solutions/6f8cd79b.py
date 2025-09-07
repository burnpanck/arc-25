def solution(input: Image) -> AnyImage:
    all = mask_all(input)
    outline = ~erode(all)
    output = paint(input, CYAN, clip=outline)
    return output
