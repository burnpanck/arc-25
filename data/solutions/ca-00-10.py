def solution(input: Canvas) -> Canvas:
    # find lower half by considering the black parts
    # as objects
    lower = max(
        find_objects(mask_color(input, BLACK)),
        key = lambda obj:center_of_mass(obj).row
    )
    c, = most_common_colors(input, exclude=BLACK)
    output = fill(input, c, clip=lower)
    return output
