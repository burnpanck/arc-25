def solution(input: Image) -> Image:
    # identify passages by color
    free = mask_color(input, BLACK)
    # identify path colors; one is next to passages
    start_color, = most_common_colors(
        apply_mask(input, dilate(free)),
        exclude={BLACK,CYAN},
    )
    # the other is the remaining non-wall, non-passage color
    other_color, = most_common_colors(
        input,
        exclude={BLACK,CYAN,start_color},
    )
    # keep track of the alternating color pattern
    pattern = [other_color,start_color]
    # start the output from the input
    output = input
    # do BFS, starting from the cells next to passages
    cur = mask_color(input, start_color)
    # BFS: keep track of "current" points from where to search
    while cur.any():
        # next points are in free space, and neighbouring `cur`
        cur = free & dilate(cur)
        # fill those with the current pattern colour
        output = fill(output,pattern[0],clip=cur)
        # roll color pattern
        pattern = pattern[::-1]
        # and mark just colored cells as non-free
        free = free & ~cur
    return output
