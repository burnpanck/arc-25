def solution(input: Canvas) -> Canvas:
    fg_cells = ~mask_color(input, BLUE)
    bbox = determine_bbox(fg_cells)
    output = extract_image(input, bbox, mask=fg_cells)
    output = fill(output, style=BLACK, clip=mask_unpainted(output))
    return output
