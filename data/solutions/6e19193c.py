def solution(input: Canvas) -> Canvas:
    fg_mask = ~mask_color(input, BLACK)
    output = input
    for obj in find_objects(fg_mask):
        bbox = find_bbox(obj)
        c, = most_common_colors(apply_mask(input, obj))
        center = center_of_mass(bbox)
        vec = center - center_of_mass(obj)
        dir = vec2dir8(vec)
        start = round2grid(center+1.5*Vector.elementary_vector(dir))
        path = path_ray(start, dir, shape=output.shape)
        output = stroke(output, path, c)
    return output
