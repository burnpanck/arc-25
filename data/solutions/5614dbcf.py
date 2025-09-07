def solution(input: Image) -> AnyImage:
    # prepare output canvas of size 3x3
    output = make_canvas((3,3))
    # iterate over each 3x3 tile
    for row in range(3):
        for col in range(3):
            # compute source tile rect
            rect = Rect.make(top=row*3,left=col*3,square_size=3)
            # determine tile color (rect->mask->most_common_colors)
            mask = rect_to_mask(input, rect)
            c, = most_common_colors(apply_mask(input,mask))
            # paint single cell using stroke
            output = stroke(output,[Coord(row,col)],c)
    return output
