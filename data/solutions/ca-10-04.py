def solution(input: Canvas) -> Canvas:
    bg = BLACK
    # find closed lines and the single magenta cell
    closed_lines = []
    cell = None
    # iterate over all objects (connectivity = 8)
    for obj in find_objects(input,connectivity=8,exclude=bg):
        # determine object color
        c, = most_common_colors(apply_mask(input, obj))
        if c == MAGENTA:
            # if magenta, must be the single cell
            cell = obj
            assert cell.count() == 1
            continue
        # otherwise, must be a closed line;
        # fill all holes and retain
        closed_lines.append(fill_holes(obj))
    # determine how many lines enclose the magenta cell
    # by counting the overlap with the filled masks
    n_enclosing = sum((obj&cell).count() for obj in closed_lines)
    if n_enclosing >= 2:
        # if two enclosing, reproduce input verbatim
        return input
    # remove magenta cell (paint background over)
    output = fill(input, BLACK, clip=cell)
    return output
