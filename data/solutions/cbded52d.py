def solution(input: Canvas) -> Canvas:
    import itertools
    markers = ~mask_color(input, {BLACK, BLUE})
    output = input
    never = mask_none(output)
    for row1 in range(2):
        for col1 in range(2):
            base = Coord(row1, col1)
            for maindir, auxdir in itertools.permutations([
                Vector.RIGHT,
                Vector.DOWN,
            ]):
                for offs in range(3):
                    start = base + 3*offs*auxdir
                    path = [start + 3*k*maindir for k in range(3)]
                    pmask = path_to_mask(output, path)
                    mmask = pmask & markers
                    if mmask.count()>=2:
                        c, = most_common_colors(apply_mask(input, mmask))
                        output = fill(output, c, clip=pmask)

    return output
