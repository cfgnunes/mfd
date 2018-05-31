import cv2


class FeatureMatching:
    def __init__(self, detector_algorithm,
                 descriptor_algorithm, max_ratio=0.8):
        '''
        The 'max_ratio' is the nearest neighbor distance ratio (NNDR)
        and is a value in the range (0,1].
        Use this value for rejecting ambiguous matches.
        Increase this value to return more matches.

        This ratio was defined in the paper [1]:
        "...we reject all matches in which the distance ratio is
        greater than 0.8, which eliminates 90% of the false matches
        while discarding less than 5% of the correct matches..."
        '''

        self.max_ratio = max_ratio
        self.detector_algorithm = detector_algorithm
        self.descriptor_algorithm = descriptor_algorithm

    def match_feature(self, image_a_filename, image_b_filename):
        # Load the images
        image_a = cv2.imread(image_a_filename, cv2.IMREAD_GRAYSCALE)
        image_b = cv2.imread(image_b_filename, cv2.IMREAD_GRAYSCALE)

        # Step 1: Detect keypoints
        keypoints_a = self.detector_algorithm.detect(image=image_a)
        keypoints_b = keypoints_a

        # Step 2: Calculate descriptors for each keypoint
        _, descriptors_a = self.descriptor_algorithm.compute(image_a,
                                                             keypoints_a)
        _, descriptors_b = self.descriptor_algorithm.compute(image_b,
                                                             keypoints_b)

        # Step 3: Match the features
        matcher_algorithm = cv2.BFMatcher_create(normType=cv2.NORM_L2)

        matches = matcher_algorithm.knnMatch(
            queryDescriptors=descriptors_a,
            trainDescriptors=descriptors_b,
            k=2)

        # Matching strategy: Nearest Neighbour Distance Ratio (NNDR),
        # defined in the paper [1]. This ratio select the best matches.
        best_matches = self.__max_ratio_test(matches, self.max_ratio)

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
        precision = self.__compute_precision(keypoints_a,
                                             keypoints_b,
                                             best_matches)

        return (image_matches, len(best_matches), precision)

    def __max_ratio_test(self, matches, max_ratio):
        best_matches = []

        for match in matches:

            if len(match) < 2:
                continue

            first_best, second_best = match

            if first_best.distance < max_ratio * second_best.distance:
                best_matches.append(first_best)

        return best_matches

    def __compute_precision(self, keypoints_a, keypoints_b, matches):
        total_matches = len(matches)

        if total_matches == 0:
            return 0.0

        correct_matches = self.__compute_correct_matches(keypoints_a,
                                                         keypoints_b,
                                                         matches)
        precision = correct_matches / total_matches

        if precision > 1.0:
            precision = 1.0

        return precision

    def __compute_correct_matches(self, keypoints_a, keypoints_b, matches):
        correct_matches = 0

        if len(matches) == 0:
            return correct_matches

        for match in matches:
            keypoint_a = keypoints_a[match.queryIdx].pt
            keypoint_b = keypoints_b[match.trainIdx].pt

            distance = cv2.norm(keypoint_a, keypoint_b, cv2.NORM_L2)

            if distance < 5.0:
                correct_matches += 1

        return correct_matches


'''
References:

[1] Lowe, David G. "Distinctive image features from scale-invariant keypoints."
International journal of computer vision 60.2 (2004): 91-110.
'''
