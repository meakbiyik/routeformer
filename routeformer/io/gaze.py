"""From PupilLabs with minor modifications.
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)

Fixations general knowledge from literature review
    + Goldberg et al. - fixations rarely < 100ms and range between 200ms and 400ms in
      duration (Irwin, 1992 - fixations dependent on task between 150ms - 600ms)
    + Very short fixations are considered not meaningful for studying behavior
        - eye+brain require time for info to be registered (see Munn et al. APGV, 2008)
    + Fixations are rarely longer than 800ms in duration
        + Smooth Pursuit is exception and different motif
        + If we do not set a maximum duration, we will also detect smooth pursuit (which
          is acceptable since we compensate for VOR)
Terms
    + dispersion (spatial) = how much spatial movement is allowed within one fixation
      (in visual angular degrees or pixels)
    + duration (temporal) = what is the minimum time required for gaze data to be within
      dispersion threshold?

"""
import abc
import copy
import enum
import logging
import typing as T
from collections import deque

import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class FixationDetectionMethod(enum.Enum):
    GAZE_2D = "2d gaze"
    GAZE_3D = "3d gaze"


def vector_dispersion(vectors):
    distances = pdist(vectors, metric="cosine")
    dispersion = np.arccos(1.0 - distances.max())
    return dispersion


def gaze_dispersion(capture, gaze_subset, method: FixationDetectionMethod) -> float:
    if method is FixationDetectionMethod.GAZE_3D:
        vectors = np.array([gp["gaze_point_3d"] for _, gp in gaze_subset])
    elif method is FixationDetectionMethod.GAZE_2D:
        locations = np.array([gp["norm_pos"] for _, gp in gaze_subset])

        # denormalize
        width, height = capture["frame_size"]
        locations[:, 0] *= width
        locations[:, 1] = (1.0 - locations[:, 1]) * height

        # undistort onto 3d plane
        vectors = capture["intrinsics"].unprojectPoints(locations)
    else:
        raise ValueError(f"Unknown method '{method}'")

    dist = vector_dispersion(vectors)
    return dist


def detect_fixations(  # noqa: C901
    capture,
    gaze_data,
    max_dispersion=np.deg2rad(1.50),
    min_duration=80 / 1000,
    max_duration=1000 / 1000,
    min_data_confidence=0.6,
):
    """Detect fixations in gaze data.

    capture: Dict with attributes:
        - frame_size: Tuple[int, int]
        - intrinsics: PupilLabs.Intrinsics
        - timestamps: np.ndarray
    gaze_data: List[Serialized_Dict]
    max_dispersion: float
        Maximum dispersion allowed for a fixation.
    min_duration: float
        Minimum duration for a fixation.
    """
    capture = copy.deepcopy(capture)
    capture["intrinsics"] = Radial_Dist_Camera._from_raw_intrinsics(
        "dummy", capture["frame_size"], capture["intrinsics"]["(1088, 1080)"]
    )
    gaze_data = [(idx, datum) for idx, datum in enumerate(gaze_data)]
    is_fixation = np.zeros(len(gaze_data), dtype=bool)
    filtered_gaze_data = [
        (idx, datum) for idx, datum in gaze_data if datum["confidence"] > min_data_confidence
    ]
    if not filtered_gaze_data:
        logger.warning("No data available to find fixations")
        return "Fixation detection failed", ()

    method = FixationDetectionMethod.GAZE_2D
    logger.info(f"Starting fixation detection using {method.value} data...")

    working_queue = deque()
    remaining_gaze = deque(filtered_gaze_data)

    while remaining_gaze:
        # check if working_queue contains enough data
        if (
            len(working_queue) < 2
            or (working_queue[-1][1]["timestamp"] - working_queue[0][1]["timestamp"]) < min_duration
        ):
            idx_datum_tuple = remaining_gaze.popleft()
            working_queue.append(idx_datum_tuple)
            continue

        # min duration reached, check for fixation
        dispersion = gaze_dispersion(capture, working_queue, method)
        if dispersion > max_dispersion:
            # not a fixation, move forward
            working_queue.popleft()
            continue

        left_idx = len(working_queue)

        # minimal fixation found. collect maximal data
        # to perform binary search for fixation end
        while remaining_gaze:
            datum = remaining_gaze[0][1]
            if datum["timestamp"] > working_queue[0][1]["timestamp"] + max_duration:
                break  # maximum data found
            working_queue.append(remaining_gaze.popleft())

        # check for fixation with maximum duration
        dispersion = gaze_dispersion(capture, working_queue, method)
        if dispersion <= max_dispersion:
            for idx, _ in working_queue:
                is_fixation[idx] = True
            working_queue.clear()  # discard old Q
            continue

        slicable = list(working_queue)  # deque does not support slicing
        right_idx = len(working_queue)

        # binary search
        while left_idx < right_idx - 1:
            middle_idx = (left_idx + right_idx) // 2
            dispersion = gaze_dispersion(
                capture,
                slicable[: middle_idx + 1],
                method,
            )
            if dispersion <= max_dispersion:
                left_idx = middle_idx
            else:
                right_idx = middle_idx

        # left_idx-1 is last valid base datum
        final_base_data = slicable[:left_idx]
        to_be_placed_back = slicable[left_idx:]
        # dispersion_result = gaze_dispersion(capture, final_base_data, method)

        for idx, _ in final_base_data:
            is_fixation[idx] = True
        working_queue.clear()  # clear queue
        remaining_gaze.extendleft(reversed(to_be_placed_back))

    logger.info(f"Found {is_fixation.sum()} fixations out of {len(is_fixation)} samples")

    return is_fixation


class RawIntrinsics(TypedDict):
    dist_coefs: T.List[T.List[float]]
    camera_matrix: T.List[T.List[float]]
    cam_type: str


RawIntrinsicsByResolution = T.Dict[str, T.Union[int, RawIntrinsics]]
RawIntrinsicsByResolutionByName = T.Dict[str, RawIntrinsicsByResolution]


class Camera_Model(abc.ABC):
    cam_type: T.ClassVar[str]  # overwrite in subclasses, used for saving/loading

    def __init__(
        self,
        name: str,
        resolution: T.Tuple[int, int],
        K: npt.ArrayLike,
        D: npt.ArrayLike,
    ):
        self.name = name
        self.resolution = resolution
        self.K: npt.NDArray[np.float64] = np.array(K)
        self.D: npt.NDArray[np.float64] = np.array(D)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} {self.name} @ "
            f"{self.resolution[0]}x{self.resolution[1]} "
            f"K={self.K.tolist()} D={self.D.tolist()}>"
        )

    def update_camera_matrix(self, camera_matrix: npt.ArrayLike):
        self.K = np.asanyarray(camera_matrix).reshape(self.K.shape)

    def update_dist_coefs(self, dist_coefs: npt.ArrayLike):
        self.D = np.asanyarray(dist_coefs).reshape(self.D.shape)

    @property
    def focal_length(self) -> float:
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        return (fx + fy) / 2

    subclass_by_cam_type: T.Dict[str, T.Type["Camera_Model"]] = dict()

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        # register subclasses by cam_type
        if not hasattr(cls, "cam_type"):
            raise NotImplementedError("Subclass needs to define 'cam_type'!")
        if cls.cam_type in Camera_Model.subclass_by_cam_type:
            raise ValueError(
                f"Error trying to register camera model {cls}: Camera model with"
                f" cam_type '{cls.cam_type}' already exists:"
                f" {Camera_Model.subclass_by_cam_type[cls.cam_type]}"
            )
        Camera_Model.subclass_by_cam_type[cls.cam_type] = cls

    def _from_raw_intrinsics(
        cam_name: str, resolution: T.Tuple[int, int], intrinsics: RawIntrinsics
    ):
        cam_type = intrinsics["cam_type"]
        if cam_type not in Camera_Model.subclass_by_cam_type:
            logger.warning(
                f"Trying to load unknown camera type intrinsics: {cam_type}! Using "
                " dummy intrinsics!"
            )
            return Dummy_Camera(cam_name, resolution)

        camera_model_class = Camera_Model.subclass_by_cam_type[cam_type]
        return camera_model_class(
            cam_name, resolution, intrinsics["camera_matrix"], intrinsics["dist_coefs"]
        )


class Radial_Dist_Camera(Camera_Model):
    """
    Camera model assuming a lense with radial distortion (this is the defaut model in
    opencv). Provides functionality to make use of a pinhole camera model that is also
    compensating for lense distortion
    """

    cam_type = "radial"

    def undistort(self, img):
        """
        Undistortes an image based on the camera model.
        :param img: Distorted input image
        :return: Undistorted image
        """
        undist_img = cv2.undistort(img, self.K, self.D)
        return undist_img

    def unprojectPoints(self, pts_2d, use_distortion=True, normalize=False):
        """
        Undistorts points according to the camera model.
        :param pts_2d, shape: Nx2
        :return: Array of unprojected 3d points, shape: Nx3
        """
        pts_2d = np.array(pts_2d, dtype=np.float32)

        # Delete any posibly wrong 3rd dimension
        if pts_2d.ndim == 1 or pts_2d.ndim == 3:
            pts_2d = pts_2d.reshape((-1, 2))

        # Add third dimension the way cv2 wants it
        if pts_2d.ndim == 2:
            pts_2d = pts_2d.reshape((-1, 1, 2))

        if use_distortion:
            _D = self.D
        else:
            _D = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0]])

        pts_2d_undist = cv2.undistortPoints(pts_2d, self.K, _D)

        pts_3d = cv2.convertPointsToHomogeneous(pts_2d_undist)
        pts_3d.shape = -1, 3

        if normalize:
            pts_3d /= np.linalg.norm(pts_3d, axis=1)[:, np.newaxis]

        return pts_3d

    def projectPoints(self, object_points, rvec=None, tvec=None, use_distortion=True):
        """
        Projects a set of points onto the camera plane as defined by the camera model.
        :param object_points: Set of 3D world points
        :param rvec: Set of vectors describing the rotation of the camera when recording
            the corresponding object point
        :param tvec: Set of vectors describing the translation of the camera when
            recording the corresponding object point
        :return: Projected 2D points
        """
        input_dim = object_points.ndim

        object_points = object_points.reshape((1, -1, 3))

        if rvec is None:
            rvec = np.zeros(3).reshape(1, 1, 3)
        else:
            rvec = np.array(rvec).reshape(1, 1, 3)

        if tvec is None:
            tvec = np.zeros(3).reshape(1, 1, 3)
        else:
            tvec = np.array(tvec).reshape(1, 1, 3)

        if use_distortion:
            _D = self.D
        else:
            _D = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0]])

        image_points, jacobian = cv2.projectPoints(object_points, rvec, tvec, self.K, _D)

        if input_dim == 2:
            image_points.shape = (-1, 2)
        elif input_dim == 3:
            image_points.shape = (-1, 1, 2)
        return image_points

    def undistort_points_to_ideal_point_coordinates(self, points):
        return cv2.undistortPoints(points, self.K, self.D)

    def solvePnP(
        self,
        uv3d,
        xy,
        flags=cv2.SOLVEPNP_ITERATIVE,
        useExtrinsicGuess=False,
        rvec=None,
        tvec=None,
    ):
        try:
            uv3d = np.reshape(uv3d, (1, -1, 3))
        except ValueError:
            raise ValueError("uv3d is not 3d points")
        try:
            xy = np.reshape(xy, (1, -1, 2))
        except ValueError:
            raise ValueError("xy is not 2d points")
        if uv3d.shape[1] != xy.shape[1]:
            raise ValueError("the number of 3d points and 2d points are not the same")

        res = cv2.solvePnP(
            uv3d,
            xy,
            self.K,
            self.D,
            flags=flags,
            useExtrinsicGuess=useExtrinsicGuess,
            rvec=rvec,
            tvec=tvec,
        )
        return res


class Dummy_Camera(Radial_Dist_Camera):
    """
    Dummy Camera model assuming no lense distortion and idealized camera intrinsics.
    """

    cam_type = "dummy"

    def __init__(self, name, resolution, K=None, D=None):
        camera_matrix = K or [
            [1000, 0.0, resolution[0] / 2.0],
            [0.0, 1000, resolution[1] / 2.0],
            [0.0, 0.0, 1.0],
        ]
        dist_coefs = D or [[0.0, 0.0, 0.0, 0.0, 0.0]]
        super().__init__(name, resolution, camera_matrix, dist_coefs)
