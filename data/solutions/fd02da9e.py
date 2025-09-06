def solution(input: Canvas, examples: list[IOPair]) -> Canvas:
    output = make_canvas(input.shape)
    output = fill(output, ORANGE)
    keys = ~mask_color(input, ORANGE)
    for idx,io in enumerate(examples):
        if io.output is None:
            continue
        key = ~mask_color(io.input, ORANGE)
        value = ~mask_color(io.output, ORANGE)
        if (keys & key == key).all():
            c, = most_common_colors(apply_mask(input, key))
            output = fill(output, c, clip=value)
    return output
