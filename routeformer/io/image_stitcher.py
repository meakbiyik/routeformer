"""Image stitcher to stitch images together from different views."""
import logging
from typing import Optional, Tuple

import cv2
import kornia
import torch
from kornia.geometry.transform import warp_perspective

logger = logging.getLogger(__name__)


class CV2RANSAC(torch.nn.Module):
    """kornia-compliant RANSAC implementation using cv2."""

    def __init__(self, method=cv2.USAC_MAGSAC, **findHomographyKwargs):
        """Initialize the RANSAC module."""
        super().__init__()

        self.method = method
        self.findHomographyKwargs = (
            findHomographyKwargs
            if findHomographyKwargs
            else {
                "ransacReprojThreshold": 0.25,
                "maxIters": 10000,
                "confidence": 0.9995,
            }
        )
        self._dummy_param = torch.nn.Parameter(torch.empty(0))

    def forward(self, src_kp: torch.Tensor, dest_kp: torch.Tensor) -> torch.Tensor:
        """Find the homography matrix using RANSAC."""
        src_kp = src_kp.cpu().numpy()
        dest_kp = dest_kp.cpu().numpy()

        logger.info(
            f"Finding homography matrix using cv2 with method {self.method} "
            f"and params {self.findHomographyKwargs}"
        )

        matrix, _ = cv2.findHomography(
            src_kp, dest_kp, method=self.method, **self.findHomographyKwargs
        )

        logger.debug(f"Found homography matrix: {matrix}")

        return (
            torch.from_numpy(matrix).to(self._dummy_param.device, dtype=torch.float32),
            None,
        )


class ImageStitcher(kornia.contrib.ImageStitcher):
    """Image stitcher module that can admit a homography matrix."""

    def __init__(self, matcher, estimator: str = "ransac", blending_method: str = "naive") -> None:
        """Initialize the image stitcher module."""
        super().__init__(matcher, estimator, blending_method)

        # Use the cv2 RANSAC implementation
        self.ransac = CV2RANSAC()

    def stitch_pair(
        self,
        images_left: torch.Tensor,
        images_right: torch.Tensor,
        mask_left: Optional[torch.Tensor] = None,
        mask_right: Optional[torch.Tensor] = None,
        prev_homography_matrix=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stitch a pair of images, estimate homography if not given."""
        # Compute the transformed images
        input_dict = self.preprocess(images_left, images_right)
        # Instead of the original implementation where the left image is padded
        # with a zero tensor size of right image horizontally, we just use
        # the original left tensor and assume that it is padded.
        out_shape = (images_left.shape[-2], images_left.shape[-1])

        if prev_homography_matrix is None:
            correspondences = self.on_matcher(input_dict)
            homo = self.estimate_transform(**correspondences)
        else:
            homo = prev_homography_matrix
        src_img = warp_perspective(images_right, homo, out_shape)
        dst_img = images_left

        # Compute the transformed masks
        if mask_left is None:
            mask_left = torch.ones_like(images_left)
        if mask_right is None:
            mask_right = torch.ones_like(images_right)
        src_mask = warp_perspective(mask_right, homo, out_shape)
        dst_mask = mask_left

        # Also refine the mask so that completely black segments
        # due to the warping are not overlaid on top of the other image
        logger.info(f"Refining the mask for image with shape {src_img.shape}")
        src_mask = torch.where(
            (src_mask < 0.75)
            | (src_img.sum((0, 1), keepdim=True).broadcast_to(src_mask.shape) < 3e-2),
            0,
            1,
        )

        return (
            self.blend_image(src_img, dst_img, src_mask),
            (dst_mask + src_mask).bool().to(src_mask.dtype),
            homo,
        )

    def forward(
        self,
        *imgs: torch.Tensor,
        homography_matrices: Optional[list[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        """Stitch given input images.

        If a list of homography matrices size len(imgs)-1 is given,
        they are used instead of estimating the homography between each
        pair.
        """
        img_out = imgs[0]
        mask_left = torch.ones_like(img_out)

        if homography_matrices is None:
            homography_matrices = [None] * (len(imgs) - 1)
        new_homography_matrices = []

        for i in range(len(imgs) - 1):
            img_out, mask_left, homo = self.stitch_pair(
                img_out,
                imgs[i + 1],
                mask_left,
                prev_homography_matrix=homography_matrices[i],
            )
            new_homography_matrices.append(homo)

        return self.postprocess(img_out, mask_left), new_homography_matrices
