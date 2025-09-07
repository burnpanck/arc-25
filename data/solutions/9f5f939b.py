def solution(input: Image) -> AnyImage:
    cells = []
    for i in [0,1,5,6]:
        cells.append(Coord(3,i))
        cells.append(Coord(i,3))
    crosshair = path_to_mask((7,7),cells)
    blues = mask_color(input, BLUE)
    centres = correlate_masks(blues, crosshair)
    output = fill(input,YELLOW,clip=centres)
    return output
