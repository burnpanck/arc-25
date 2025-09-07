def solution(input: Image) -> AnyImage:
    # find gray objects and sort them by their cell count
    objs = sorted(
        find_objects(mask_color(input, GRAY)),
        key=lambda objmsk:objmsk.count(),
    )
    output = input
    # iterate ober gray objects, and color them red,yellow and blue, in turn
    for obj,c in zip(objs, [RED,YELLOW,BLUE]):
        output = fill(output, c, clip=obj)
    return output
