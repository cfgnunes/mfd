import cv2

_PIXEL_ERROR = 5.0


class FeatureMatching:

    def __init__(self, detector_alg, descriptor_alg, nndr_ratio=0.8):
        self._detector_alg = detector_alg
        self._descriptor_alg = descriptor_alg
        self._nndr_ratio = nndr_ratio

    def match_features(self, image_a_filename, image_b_filename):
        # Load the images
        image_a = cv2.imread(image_a_filename, cv2.IMREAD_GRAYSCALE)
        image_b = cv2.imread(image_b_filename, cv2.IMREAD_GRAYSCALE)

        # Step 1: Detect keypoints
        keypoints_a = self._detector_alg.detect(image=image_a)
        keypoints_b = keypoints_a

        # Step 2: Compute descriptors for each keypoint
        _, descriptors_a = self._descriptor_alg.compute(image_a, keypoints_a)
        _, descriptors_b = self._descriptor_alg.compute(image_b, keypoints_b)

        # Step 3: Compute the matches
        matcher_algorithm = cv2.BFMatcher_create(normType=cv2.NORM_L2)

        matches = matcher_algorithm.knnMatch(
            queryDescriptors=descriptors_a,
            trainDescriptors=descriptors_b,
            k=2)

        # Matching strategy: Nearest Neighbour Distance Ratio (NNDR),
        # defined in the paper [1]. This ratio select the best matches.
        best_matches = self.nearest_neighbor_test(matches, self._nndr_ratio)

        # Plot matching result
        image_matches = cv2.drawMatches(
            img1=image_a,
            keypoints1=keypoints_a,
            img2=image_b,
            keypoints2=keypoints_b,
            matches1to2=best_matches,
            outImg=0,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Compute precision
        precision = self.compute_precision(
            keypoints_a, keypoints_b, best_matches)

        return (image_matches, len(best_matches), precision)

    @staticmethod
    def nearest_neighbor_test(matches, nndr_ratio):
        '''
        The 'nndr_ratio' is the nearest neighbor distance ratio (NNDR)
        and is a value in the range (0,1].
        Use this value for rejecting ambiguous matches.
        Increase this value to return more matches.

        This ratio was defined in the paper [1]:
        "...we reject all matches in which the distance ratio is
        greater than 0.8, which eliminates 90% of the false matches
        while discarding less than 5% of the correct matches..."
        '''

        best_matches = []

        if matches is None:
            return None

        for match in matches:
            if len(match) < 2:
                continue

            first_best, second_best = match

            if first_best.distance < nndr_ratio * second_best.distance:
                best_matches.append(first_best)

        return best_matches

    @staticmethod
    def compute_precision(keypoints_a, keypoints_b, matches):
        num_matches = len(matches)

        if num_matches == 0:
            return 0.0

        correct_matches, _ = FeatureMatching.get_correct_matches(
            keypoints_a, keypoints_b, matches)
        num_correct_matches = len(correct_matches)

        precision = num_correct_matches / num_matches

        if precision > 1.0:
            precision = 1.0

        return precision

    @staticmethod
    def get_correct_matches(keypoints_a, keypoints_b, matches):
        correct_matches = []
        incorrect_matches = []

        if not matches:
            return correct_matches, incorrect_matches

        for match in matches:
            point_a = keypoints_a[match.queryIdx].pt
            point_b = keypoints_b[match.trainIdx].pt

            if FeatureMatching.is_region_overlap(point_a, point_b):
                correct_matches.append(match)
            else:
                incorrect_matches.append(match)

        return correct_matches, incorrect_matches

    @staticmethod
    def is_region_overlap(point_a, point_b):
        error = cv2.norm(point_a, point_b, cv2.NORM_L2)
        if error < _PIXEL_ERROR:
            return True

        return False


# References:
#
# [1] Lowe, David G. "Distinctive image features from scale-invariant
# keypoints." International journal of computer vision 60.2 (2004): 91-110.
