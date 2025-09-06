def solution(input: Canvas) -> Canvas:
    output = input
    # iterate overa all objects
    for obj in find_objects(input, exclude=BLACK):
        # determine interior by hole filling
        filled = fill_holes(obj)
        # skip if there is no interior
        if (filled == obj).all():
            continue
        # determine border mask (both internal and external combined)
        border = dilate(obj, connectivity=8) & ~obj
        # the intersection with
        output = fill(output, RED, clip=border & ~filled)
        output = fill(output, GREEN, clip=border & filled)

    return output
