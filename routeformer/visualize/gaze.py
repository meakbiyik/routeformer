"""Gaze visualization methods."""
import cv2
import numpy as np


def overlay_heatmap_on_frame(frame, gaze_points, colormap=cv2.COLORMAP_JET):
    """Overlays heatmap of gaze points on the frame.

    Parameters
    ----------
    frame : np.ndarray
        A frame from a video, with shape (height, width, channels), in
        BGR format with range [0, 255].
    gaze_points : np.ndarray
        Gaze points in the frame, with shape (num_points, 2). The points are
        normalized to the frame size, i.e. (0, 0) is the bottom left corner
        and (1, 1) is the top right corner, with order (width, height).
    colormap : int
        OpenCV colormap to use for the heatmap. Default: cv2.COLORMAP_JET.

    Returns
    -------
    overlaid_frame : np.ndarray
        The frame with the gaze heatmap overlaid on it.
    """
    heatmap = _draw_gaussian_at_point(frame, gaze_points)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    colored_heatmap = cv2.applyColorMap(np.uint8(heatmap), colormap)
    overlaid_frame = cv2.addWeighted(frame, 0.6, colored_heatmap, 0.4, 0)
    # only choose the heatmap pixels that are above a certain threshold
    extended_heatmap = np.expand_dims(heatmap, axis=-1)
    overlaid_frame = np.where(extended_heatmap > 0.2, overlaid_frame, frame)
    return overlaid_frame


def _draw_gaussian_at_point(frame, points, sigma=10, amplitude=255):
    """Draws a Gaussian kernel at a specified point in the image."""
    heatmap = np.zeros_like(frame[:, :, 0], dtype=np.float32)
    rows, cols = heatmap.shape
    for y, x in points:
        # point: (x, y) = (normalized_height, normalized_width)
        # relative to the bottom left corner of the frame
        x, y = int((1 - x) * cols), int(y * rows)
        meshgrid = np.meshgrid(np.arange(cols), np.arange(rows))
        heatmap += amplitude * np.exp(
            -((meshgrid[0] - y) ** 2 + (meshgrid[1] - x) ** 2) / (2 * sigma**2)
        )
    return heatmap
