"""Feature Matching.

Implementation by: Cristiano Fraga G. Nunes <cfgnunes@gmail.com>

References:
[1] https://dx.doi.org/10.1023/b:visi.0000029664.99615.94
"""

import random
import cv2


class FeatureMatching:
    """Feature Matching."""

    def __init__(self, detector_alg, descriptor_alg,
                 nndr_ratio=0.8, max_features=None):
        """Class constructor.

        Args:
            detector_alg (cv2.Feature2D): detector algorithm.
            descriptor_alg (cv2.Feature2D): descriptor algorithm.
            nndr_ratio (float, optional): the Nearest Neighbor Distance Ratio.
                Defaults to 0.8.
            max_features (int, optional): the maximum number of detected
                features. Defaults to None.
        """
        self._detector_alg = detector_alg
        self._descriptor_alg = descriptor_alg
        self._nndr_ratio = nndr_ratio
        self._max_features = max_features

    def match_features(self, image_a, image_b):
        """Compute the matches of the features.

        Args:
            image_a (numpy.ndarray): input image A.
            image_b (numpy.ndarray): input image B.

        Returns:
            tuple of:
                numpy.ndarray: keypoints_a
                numpy.ndarray: keypoints_b
                numpy.ndarray: descriptors_a
                numpy.ndarray: descriptors_b
                tuple of ((cv2.DMatch, cv2.DMatch),): matches
        """
        matches = ()

        # Step 1: Detect keypoints
        keypoints_a = self._get_keypoints(image=image_a)
        keypoints_b = self._get_keypoints(image=image_b)

        # Step 2: Compute descriptors for each keypoint
        descriptors_a = self._get_descriptors(image_a, keypoints_a)
        descriptors_b = self._get_descriptors(image_b, keypoints_b)

        # Step 3: Compute the matches
        matches = self._get_matches(descriptors_a, descriptors_b)

        return keypoints_a, keypoints_b, descriptors_a, descriptors_b, matches

    def match_features_evaluation(self, image_a, image_b, transformation=None):
        """Compute the matches of the features for evaluation.

        Detect keypoints only in the image A and project them into image B.

        Args:
            image_a (numpy.ndarray): input image A.
            image_b (numpy.ndarray): input image B.
            transformation (ImageTransformation): the image transformation
                object. Defaults to None.

        Returns:
            tuple of:
                numpy.ndarray: keypoints_a
                numpy.ndarray: keypoints_b
                numpy.ndarray: descriptors_a
                numpy.ndarray: descriptors_b
                tuple of ((cv2.DMatch, cv2.DMatch),): matches
        """
        matches = ()

        # Step 1: Detect keypoints
        keypoints_a = self._get_keypoints(image=image_a)

        # Step 1.5: Apply a transformation in the image B and
        #           project keypoints from image A to image B
        if transformation:
            image_b = transformation.transform_image(image_b)
            keypoints_b = transformation.transform_keypoints(
                image_a, keypoints_a)
        else:
            keypoints_b = keypoints_a

        # Step 2: Compute descriptors for each keypoint
        descriptors_a = self._get_descriptors(image_a, keypoints_a)
        descriptors_b = self._get_descriptors(image_b, keypoints_b)

        # Step 3: Compute the matches
        matches = self._get_matches(descriptors_a, descriptors_b)

        return keypoints_a, keypoints_b, descriptors_a, descriptors_b, matches

    def _get_keypoints(self, image):
        if image is None:
            return None

        keypoints = self._detector_alg.detect(image)

        if self._max_features:
            random.seed(0)
            keypoints = tuple(random.sample(keypoints, len(keypoints)))
            keypoints = keypoints[:self._max_features]

        return keypoints

    def _get_descriptors(self, image, keypoints):
        if keypoints is None or image is None:
            return None

        if not keypoints:
            return None

        _, descriptors = self._descriptor_alg.compute(image, keypoints)
        return descriptors

    def _get_matches(self, descriptors_a, descriptors_b):
        if descriptors_a is None or descriptors_b is None:
            return None

        if not descriptors_a.size or not descriptors_b.size:
            return None

        matcher_alg = cv2.BFMatcher_create(normType=cv2.NORM_L2)

        matches = matcher_alg.knnMatch(
            queryDescriptors=descriptors_a,
            trainDescriptors=descriptors_b,
            k=2)

        # Matching strategy: Nearest Neighbor Distance Ratio (NNDR),
        # defined in the paper [1]. This strategy select the best matches.
        matches = self.nearest_neighbor_matches(matches, self._nndr_ratio)

        return matches

    @staticmethod
    def nearest_neighbor_matches(matches, nndr_ratio=0.8):
        """Compute the Nearest Neighbor Distance Ratio test.

        The 'nndr_ratio' is the Nearest Neighbor Distance Ratio (NNDR) and is a
        value in the range (0,1]. Use this value for rejecting ambiguous
        matches. Increase this value to return more matches.

        This ratio was defined in the paper [1]: "...we reject all matches in
        which the distance ratio is greater than 0.8, which eliminates 90% of
        the false matches while discarding less than 5% of the correct
        matches..."

        Args:
            matches (tuple of ((cv2.DMatch, cv2.DMatch),)): computed matches
                between descriptors from images A and B.
            nndr_ratio (float, optional): the Nearest Neighbor Distance Ratio
                Defaults to 0.8.

        Returns:
            tuple of ((cv2.DMatch, cv2.DMatch),): computed matches between
                descriptors from images A and B filtered by the Nearest
                Neighbor Distance Ratio test.
        """
        best_matches = ()

        if matches is None:
            return None

        if nndr_ratio > 1.0:
            for match in matches:
                first_best, _ = match
                best_matches += (first_best,)
        else:
            for match in matches:
                if len(match) < 2:
                    continue

                first_best, second_best = match

                if first_best.distance < nndr_ratio * second_best.distance:
                    best_matches += (first_best,)

        return best_matches
