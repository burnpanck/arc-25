def solution(input: Image) -> AnyImage:
    fg_mask = ~mask_color(input, BLACK)
    output = input
    # iterate over all intact object instances
    for obj in find_objects(fg_mask, connectivity=8):
        # (those consisting of more than one color)
        masked_input = apply_mask(input, obj)
        ccnt = count_colors(masked_input)
        colors = ccnt.as_set()
        if len(colors)<2:
            continue
        # extract intact object as a reference
        bbox = find_bbox(obj)
        ref = extract_image(masked_input, rect=bbox)
        # iterate over colors of the reference
        for c in colors:
            mask = mask_color(input, c)
            refmask = mask_color(ref, c)
            # find other occurences of the colored subobject
            for pos in find_cells(correlate_masks(mask, refmask)):
                # paint-over the reference
                output = paste(output, ref, center=pos)
    return output
