# first attempt:

def precursor(input: Image) -> AnyImage:
    divider = mask_color(input, ORANGE)
    area = erode(rect_to_mask(input.shape, find_bbox(divider)))
    enclosed_objs = area & ~mask_color(input, GRAY)
    bbox = find_bbox(enclosed_objs)
    output = extract_image(input, rect=bbox)
    return output

# observation: erosion cuts square at canvas edge; that fails the rule
# -> revised plan: don't erode, just exclude orange

def solution(input: Image) -> AnyImage:
    divider = mask_color(input, ORANGE)
    area = rect_to_mask(input.shape, find_bbox(divider))
    squares = ~mask_color(input, {ORANGE,GRAY})
    enclosed_objs = area & squares
    bbox = find_bbox(enclosed_objs)
    output = extract_image(input, rect=bbox)
    return output
