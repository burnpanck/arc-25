def solution(input: Canvas) -> Canvas:
    # find the center of all red squares and store those
    attractors = [
        center_of_mass(obj)
        for obj in find_objects(mask_color(input, RED))
    ]
    output = input
    # iterate over each blue cell
    for cell in find_cells(mask_color(input, BLUE)):
        # find the center of the nearest red square
        tgt = min(
            attractors,
            key = lambda tgt: (tgt - cell).manhattan(),
        )
        # figure out if that red square is horizontally
        # adjacent; determine lateral offset from axis-aligned
        # directions as the difference between the manhattan distance
        # and the chebyshev distance
        offset = tgt-cell
        lateral_offset = offset.manhattan() - offset.chebyshev()
        # because the red square is 2×2,
        # any lateral offset>1 is misaligned
        if lateral_offset>1:
            continue
        # determine axis-aligned move direction
        dir = vec2dir4(offset)
        # determine move distance (target position is 1.5 cells from the square center)
        move_distance = offset.chebyshev()-1.5
        # first paint black over original location
        # then paint blue ovver new location
        newpos = round2grid(cell + move_distance*Vector.elementary_vector(dir))
        output = stroke(output, [cell], BLACK)
        output = stroke(output, [newpos], BLUE)
    return output
