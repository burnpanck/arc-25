def solution(input: Canvas) -> Canvas:
    # repeat the analysis for both transposed and original
    # orientation, so we only have to look for vertical lines
    for op in [IDENTITY, FLIP_DIAG_MAIN]:
        # identify rectangle
        input_tfo = transform(input, op)
        fg_cells = ~mask_color(input_tfo, BLACK)
        rect = find_bbox(fg_cells)
        # test if there are any blue cells in the interior
        rect_mask = rect_to_mask(input_tfo, rect)
        interior = erode(rect_mask)
        interior_fg = interior & fg_cells
        # start by building the outline mask
        mask = rect_mask & ~interior
        if interior_fg.any():
            # if there are interior cells, look for a column
            # with at least two blue cells.
            for col in range(rect.left+1,rect.right-1):
                col_mask = mask_col(input_tfo, col)
                if not (interior_fg & col_mask).count()>1:
                    continue
                # add that column to the outline mask
                mask = mask | (col_mask & interior)
                break
            else:
                continue
        # now, paint the full mask in red
        output = fill(input_tfo, RED, clip=mask)
        # then paint over the origianl blue cells
        output = fill(output, BLUE, clip=fg_cells)
        return transform(output, op)
