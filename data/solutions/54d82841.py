def solution(input: Image) -> AnyImage:
    h,w = input.shape
    ref = mask_from_string("[xxx|xox]")
    print(ref.shape)
    fg_mask = ~mask_color(input, BLACK)
    output = input
    for pos in find_cells(correlate_masks(fg_mask, ref)):
        print(pos)
        output = stroke(output, [Coord(h-1,pos.col)], YELLOW)
    return output
