def attempt1(input: Image) -> AnyImage:
    mask = ~mask_color(input, BROWN)
    masked = apply_mask(input, mask)
    output = make_canvas(input.shape)
    io = Vector(1,1)
    for outer in range(2):
        for inner in range(2):
            if True:
                output = paste(output, masked, topleft=Coord(1,1)-io)
            masked = transform(masked, FLIP_LR)
            io = Vector(io.row,-io.col)
        masked = transform(masked, FLIP_UD)
        io = Vector(-io.row,io.col)
    return output

# we realise that some cells remain indeterminate;
# -> new rule: assume 90° symmetry as needed

def solution(input: Image) -> AnyImage:
    mask = ~mask_color(input, BROWN)
    masked = apply_mask(input, mask)
    output = make_canvas(input.shape)
    io = Vector(1,1)
    for diag in range(2):
        masked = transform(masked, FLIP_DIAG_MAIN)
        io = Vector(io.col,io.row)
        for outer in range(2):
            for inner in range(2):
                if True:
                    output = paste(output, masked, topleft=Coord(1,1)-io)
                masked = transform(masked, FLIP_LR)
                io = Vector(io.row,-io.col)
            masked = transform(masked, FLIP_UD)
            io = Vector(-io.row,io.col)
    return output
