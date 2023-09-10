"""Matching Evaluation.

Implementation by: Cristiano Fraga G. Nunes <cfgnunes@gmail.com>

The number of correct matches and correspondences are determined with the same
criterion and it uses the basic idea of 'overlap error' as described in the
paper [1] and [2]. But, instead of using 'overlap error', two regions can be
deemed to correspond if the Euclidean distance is less than 5 pixels.

References:
[1] https://dx.doi.org/10.1007/3-540-47969-4_9
[2] https://dx.doi.org/10.1109/tpami.2005.188

Terms:
matches         --> True positives + False positives.
correct_matches --> True positives.
correspondences --> True positives + False negatives (relevant elements).
"""

import cv2
import numpy as np


class MatchingEvaluation:
    """Matching Evaluation."""

    def __init__(self, keypoints_a, keypoints_b,
                 matches, homography_a_to_b=None, pixel_error=5.0):
        """Class constructor.

        Args:
            keypoints_a (numpy.ndarray): keypoints of the image A.
            keypoints_b (numpy.ndarray): keypoints of the image B.
            matches (tuple of ((cv2.DMatch, cv2.DMatch),)): computed matches
                between descriptors from images A and B.
            homography_a_to_b (numpy.ndarray): the homography matrix that maps
                keypoints from image A to the image B. Defaults to None.
            pixel_error (float, optional): pixel error to compute the overlap
                between two keypoints. Defaults to 5.0.
        """
        self._matches = matches
        self._pixel_error = pixel_error
        self._points_a = ()
        self._points_b = ()

        if keypoints_a is not None:
            self._points_a = cv2.KeyPoint_convert(keypoints_a)

        if keypoints_b is not None:
            self._points_b = cv2.KeyPoint_convert(keypoints_b)

        if self._matches is None:
            self._matches = ()

        if homography_a_to_b is not None:
            points_p = cv2.perspectiveTransform(
                self._points_a[None, :, :], homography_a_to_b)
            self._points_a = points_p[0].round()

    def compute_recall_precision(self):
        """Compute the values of recall and precision.

        Returns:
            tuple of (float, float): recall and precision values.
        """
        correct_matches, _ = self.get_correct_matches()
        n_correct_matches = len(correct_matches)
        n_matches = len(self._matches)
        n_overlaps = self.get_n_overlaps(3.0)

        recall = 0.0
        precision = 0.0

        if n_overlaps != 0:
            recall = n_correct_matches / n_overlaps

        recall = min(recall, 1.0)

        if n_matches != 0:
            precision = n_correct_matches / n_matches

        precision = min(precision, 1.0)

        return recall, precision

    def compute_repeatability(self):
        """Compute the value of repeatability.

        Returns:
            float: repeatability value.
        """
        n_overlaps = self.get_n_overlaps(self._pixel_error)
        n_points_a = len(self._points_a)
        n_points_b = len(self._points_b)

        repeatability = 0.0

        denominator = min(n_points_a, n_points_b)

        if denominator != 0:
            repeatability = n_overlaps / denominator

        return repeatability

    def get_correct_matches(self):
        """Get the correct matches and the incorrect matches.

        Returns:
            tuple of ((cv2.DMatch, cv2.DMatch),): correct matches between
                images A and B.
        """
        correct_matches = ()
        incorrect_matches = ()

        for match in self._matches:
            point_a = self._points_a[match.queryIdx]
            point_b = self._points_b[match.trainIdx]

            if self._is_points_overlaped(point_a, point_b):
                correct_matches += (match,)
            else:
                incorrect_matches += (match,)

        return correct_matches, incorrect_matches

    def get_n_overlaps(self, pixel_error):
        """Get the number of overlapped regions.

        Count how many keypoints in 'image A' has a overlapped keypoint in
        'image B'.

        Args:
            pixel_error (float): pixel error to compute the overlap
                between two keypoints.

        Returns:
            int: the number of overlapped regions.
        """
        matcher_alg = cv2.BFMatcher_create(normType=cv2.NORM_L2)
        matches_overlaps = matcher_alg.match(
            queryDescriptors=self._points_a,
            trainDescriptors=self._points_b)

        all_distances = np.array([m.distance for m in matches_overlaps])

        return np.count_nonzero(all_distances < pixel_error)

    def _is_points_overlaped(self, point_a, point_b):
        error = cv2.norm(point_a, point_b, cv2.NORM_L2)

        if error < self._pixel_error:
            return True

        return False
