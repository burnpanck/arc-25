def solution(input: Canvas) -> Canvas:
    # extract input pattern (tight bbox around non-black cells)
    fg_mask = ~mask_color(input, BLACK)
    bbox = find_bbox(fg_mask)
    pattern = extract_image(input, rect=bbox)
    # repetition spacing is bbox height
    spacing = bbox.height
    # paste input pattern in a loop
    # attention: start-row may be outside of the canvas,
    # but should be an integer multiple of the spacing
    # away from the current location
    output = input
    for row in range(bbox.top%spacing-spacing,input.shape[0],spacing):
        output = paste(output, pattern, topleft=(row,0))
    return output
