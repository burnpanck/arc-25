def solution(input: Image) -> AnyImage:
    # find the two "fields"
    bg = identify_background(input, mode="edge")
    fg_mask = ~mask_color(input, bg)
    # collect the subobjects inside the "spec field"
    # and remember the "target field"
    tgt = None
    src = []
    for field in find_objects(fg_mask):
        # search for all subobjects within the current field
        area_bbox = find_bbox(field)
        area_bg = identify_background(extract_image(input,rect=area_bbox), mode="edge")
        subobj_mask = rect_to_mask(input, area_bbox) & ~mask_color(input,area_bg)
        subobj = list(find_objects(subobj_mask))
        field_bbox = reduce_rect(area_bbox,amount=1)
        if subobj:
            # there are subobjects, so this is the "spec field"
            for o in subobj:
                ccnt = count_colors(apply_mask(input, o))
                c, = most_common_colors(ccnt)
                bbox = find_bbox(o)
                # remember if the subobject shape, color, and
                # and if it aligns left or right with the field
                src.append((
                    bbox.left <= field_bbox.left,
                    bbox.right >= field_bbox.right,
                    bbox,
                    c,
                ))
        else:
            # no subobjects, so this is the "target field"
            tgt = (field_bbox.left, field_bbox.right)
    # now, replicate the subobjects into the target field
    left,right = tgt
    output = input
    for is_left, is_right, bbox, c in src:
        is_fixed_width = not (is_left and is_right)
        r = Rect.make(
            left = left if is_left else None,
            right = right if is_right else None,
            width = bbox.width if is_fixed_width else None,
            top = bbox.top,
            bottom = bbox.bottom,
        )
        # paint rect `r` in color `c`
        output = fill(output, c, clip=rect_to_mask(output, r))
    return output
