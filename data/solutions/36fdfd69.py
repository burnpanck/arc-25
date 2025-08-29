def solution(input: Canvas) -> Canvas:
    from scipy import ndimage

    objects = mask_color(input, RED)
    structure = ndimage.iterate_structure(
        ndimage.generate_binary_structure(2,1),
        2,
    )
    print(structure.shape, objects._mask.shape)
    labeled_array, num_features = ndimage.label(
        objects._mask, structure,
    )
    ret = []
    for label in range(1, num_features + 1):
        ret.append(Mask(labeled_array == label))
    return output
