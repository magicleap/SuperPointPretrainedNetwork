import torch

# Constants
cell_size = 8  # Linear size of SuperPoint output cell

# Parameters
conf_thresh = 0.015  # Key point confidence threshold
nms_dist = 4  # Non Maximum Suppression (NMS) distance
border_padding = 4  # Remove key points this close to the border


def _nms_fast(in_keypts: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Runs a faster approximate Non-Max-Suppression (NMS) on key points.
    This function is based on `SuperPointFrontend.nms_fast()` but doesn't use NumPy -- look there for more details.

    :param in_keypts: tensor shaped 3 x N with key points [x, y, confidence].
    :param h: image height.
    :param w: image width.
    :return: tensor shaped 3 x N with surviving key points.
    """

    grid = torch.zeros(h, w, dtype=torch.int)  # Track NMS data
    inds = torch.zeros(h, w, dtype=torch.int)  # Store indices of points

    # Sort by confidence and round to the nearest int
    sorted_inds = torch.argsort(-in_keypts[2, :])
    keypts = in_keypts[:, sorted_inds]
    r_keypts = torch.round(keypts[:2, :]).int()

    # Check for edge case of 0 or 1 key points
    if r_keypts.size(dim=1) == 0:
        return torch.zeros(3, 0, dtype=torch.int)
    if r_keypts.size(dim=1) == 1:
        return torch.reshape(torch.vstack([r_keypts, in_keypts[2]]), [3, 1])

    # Initialize the grid
    for i in range(r_keypts.size(dim=1)):
        grid[r_keypts[1, i], r_keypts[0, i]] = 1
        inds[r_keypts[1, i], r_keypts[0, i]] = i

    # Pad the border of the grid, so that we can NMS points near the border
    grid = torch.nn.functional.pad(grid, [nms_dist, nms_dist, nms_dist, nms_dist], mode='constant')

    # Iterate through points, from highest to lowest conference, suppress neighborhood
    for i, r_keypt in enumerate(r_keypts.t()):
        pt = (r_keypt[0] + nms_dist, r_keypt[1] + nms_dist)  # Account for top and left padding
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed
            grid[(pt[1] - nms_dist):(pt[1] + nms_dist + 1), (pt[0] - nms_dist):(pt[0] + nms_dist + 1)] = 0
            grid[pt[1], pt[0]] = -1

    # Get all surviving -1's and return sorted array of remaining corners
    keep_y, keep_x = torch.where(grid == -1)
    keep_y, keep_x = keep_y - nms_dist, keep_x - nms_dist
    inds_keep = inds[keep_y, keep_x].long()
    out = keypts[:, inds_keep]
    sorted_inds = torch.argsort(-out[-1, :])
    out = out[:, sorted_inds]
    return out


def decode_output(semi_keypts: torch.Tensor, coarse_descrs: torch.Tensor, h: int, w: int) -> (
        torch.Tensor, torch.Tensor):
    """
    Decodes SuperPoint output extracting key points and descriptors.
    This function is based on `SuperPointFrontend.run()` but doesn't use NumPy.

    :param semi_keypts: tensor shaped N x 65 x H/8 x W/8 with raw key points.
    :param coarse_descrs: tensor shaped N x 256 x H/8 x W/8 with raw descriptors.
    :param h: image height.
    :param w: image width.
    :return: tensor shaped 3 x M with key points [x, y, confidence] and tensor shaped 256 x M with descriptors.
    """

    semi_keypts = torch.squeeze(semi_keypts)

    # --- Process points

    dense = torch.exp(semi_keypts)  # Softmax
    dense /= torch.sum(dense, dim=0) + .00001  # Should sum to 1
    no_dust = dense[:-1, :, :]  # Remove dustbin

    # Reshape to get full resolution heatmap
    h_c = int(h / cell_size)
    w_c = int(w / cell_size)
    no_dust = torch.permute(no_dust, [1, 2, 0])
    heatmap = torch.reshape(no_dust, [h_c, w_c, cell_size, cell_size])
    heatmap = torch.permute(heatmap, [0, 2, 1, 3])
    heatmap = torch.reshape(heatmap, [h_c * cell_size, w_c * cell_size])

    # Fixed coordinates names (https://github.com/magicleap/SuperPointPretrainedNetwork/pull/14)
    ys, xs = torch.where(heatmap >= conf_thresh)
    if len(xs) == 0:
        return torch.zeros(3, 0), torch.zeros(coarse_descrs.size(dim=1), 0)

    # Populate point data
    keypts = torch.zeros(3, len(xs))
    keypts[0, :] = xs
    keypts[1, :] = ys
    keypts[2, :] = heatmap[ys, xs]

    keypts = _nms_fast(keypts, h, w)  # Apply NMS

    # Sort by confidence
    sorted_inds = torch.argsort(keypts[2, :], descending=True)
    keypts = keypts[:, sorted_inds]

    # Remove points along border
    to_remove_w = torch.logical_or(keypts[0, :] < border_padding, keypts[0, :] >= (w - border_padding))
    to_remove_h = torch.logical_or(keypts[1, :] < border_padding, keypts[1, :] >= (h - border_padding))
    to_remove = torch.logical_or(to_remove_w, to_remove_h)
    keypts = keypts[:, ~to_remove]

    # --- Process descriptors

    if keypts.size(dim=1) == 0:
        return keypts, torch.zeros(coarse_descrs.size(dim=1), 0)

    # Interpolate into descriptor map using 2D point locations
    sample_keypts = keypts[:2, :].clone()
    sample_keypts[0, :] = (sample_keypts[0, :] / (w / 2)) - 1
    sample_keypts[1, :] = (sample_keypts[1, :] / (h / 2)) - 1
    sample_keypts = torch.transpose(sample_keypts, 0, 1).contiguous()
    sample_keypts = sample_keypts.view(1, 1, -1, 2)
    descrs = torch.nn.functional.grid_sample(coarse_descrs, sample_keypts, align_corners=True)
    descrs = torch.reshape(descrs, [coarse_descrs.size(dim=1), -1])
    descrs /= torch.linalg.norm(descrs, dim=0)[None, :]

    return keypts, descrs
