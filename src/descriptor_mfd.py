import cv2
import numpy as np
import phasepack


class MFD():
    '''
    Extract the MFD (Multiespectral Feature Descriptor) descriptor

    Paper [1]:
       "A Local Feature Descriptor Based on Log-Gabor Filters
       for Keypoint Matching in Multispectral Images"
    Authors:
        Cristiano Nunes <cfgnunes@gmail.com>
        Flávio Luis Cardeal Padua <cardeal@decom.cefetmg.br>

    Input image should be a single-band image,
    but if it's a multiband (e.g. RGB) image
    only the 1st band will be used.

    The output descriptor is a array with 4x4x2x5=160 values.

    The image is split into 4x4 non-overlapping rectangular regions.
    For each region, 10 (2x5) Log-Gabor filter is computed.
    '''

    _N_ROW_REGIONS = 4
    _N_COL_REGIONS = 4

    def __init__(self, window_size=80, n_scales=2, n_orientations=5):
        self.window_size = window_size
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self._filters = []
        self._nfilters = n_scales * n_orientations

    def descriptor_size(self):
        return int(self._N_ROW_REGIONS * self._N_COL_REGIONS * self._nfilters)

    @staticmethod
    def descriptor_type():
        return np.float32

    # Return a list of descriptors for each region around all keypoints.
    # The size of the region is defined by "window_size".
    def compute(self, image, keypoints):

        if not keypoints:
            return None

        images_log_gabor = self._apply_filters(image)
        maxp = self._build_map(images_log_gabor)

        descriptors = np.zeros(
            (len(keypoints), self.descriptor_size()), self.descriptor_type())

        for i, k in enumerate(keypoints):
            # Crop a region (window_size X window_size)
            # around the keypoint
            window_maxp = self._crop_region(
                maxp, (k.pt[1], k.pt[0]), self.window_size)

            if window_maxp is None:
                continue

            # Compute the descriptor for this region
            descriptors[i] = self._compute_subregions(window_maxp)

        return keypoints, descriptors

    # Return the descriptor for a image
    def compute_descriptor(self, image):
        images_log_gabor = self._apply_filters(image)
        maxp = self._build_map(images_log_gabor)
        descriptor = self._compute_subregions(maxp)
        return descriptor

    # Return a matrix containing the index values of maximum filters responses.
    # Each value of this returned matrix represent which filter had a maximum
    # response for that pixel coordinate.
    def _build_map(self, images_log_gabor):
        image_rows, image_cols = images_log_gabor[0][0].shape
        images_array = np.zeros(
            (image_rows, image_cols, self._nfilters), np.float32)

        i = 0
        for s in range(self.n_scales):
            for o in range(self.n_orientations):
                # Compute the absolute values
                images_array[:, :, i] = np.abs(images_log_gabor[o][s])
                i = i + 1

        maxp = images_array.argmax(2)
        maxp += 1

        return maxp

    # Apply the filters in the image
    def _apply_filters(self, image):
        if image.size == 0:
            return None

        if image.dtype != np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # The values of the parameters for the log-Gabor
        # filters were defined as suggested in a previous
        # study [2], because they demonstrated good results
        # in texture extraction when log-Gabor filters were
        # used for image descriptions.
        min_wavelen = 3
        scale_factor = 2
        sigma_on_f = 0.65

        # Build a filter bank with a Log-Gabor filters
        _, _, _, _, _, images_filtered, _ = phasepack.phasecong(
            image, nscale=self.n_scales, norient=self.n_orientations,
            minWaveLength=min_wavelen, mult=scale_factor, sigmaOnf=sigma_on_f,
            k=2.0, g=3)

        return images_filtered

    # Compute the histogram for each subregion and return the descriptor.
    # Every pixel of each subregion contributes to a bin on
    # the histogram according to the filters.
    def _compute_subregions(self, maxp):
        descriptor = np.zeros(self.descriptor_size(), self.descriptor_type())
        image_rows, image_cols = maxp.shape

        step = 0
        for i in range(self._N_ROW_REGIONS):
            for j in range(self._N_COL_REGIONS):
                i_1 = i * round(image_rows / self._N_ROW_REGIONS)
                j_1 = j * round(image_cols / self._N_COL_REGIONS)
                i_2 = (i + 1) * round(image_rows / self._N_ROW_REGIONS)
                j_2 = (j + 1) * round(image_cols / self._N_COL_REGIONS)

                # Crop the subregion
                subregion = maxp[i_1:i_2, j_1:j_2]

                # Compute the histogram
                histogram = cv2.calcHist(
                    images=[subregion.astype(np.uint8)], channels=None,
                    mask=None, histSize=[self._nfilters],
                    ranges=[1, self._nfilters + 1])
                descriptor[step:(step + self._nfilters)] = histogram.T
                step += self._nfilters

        # Normalize the final descriptor
        cv2.normalize(descriptor, descriptor)
        return descriptor

    # Crop a region of (size X size) centered around a center reference
    @staticmethod
    def _crop_region(image, center, size):
        image_rows, image_cols = image.shape
        i, j = center
        half_size = size / 2

        # Check the boundaries
        i_1 = np.max((0, i - half_size))
        i_1 = np.min((image_rows, i_1 + 1))
        i_2 = np.min((image_rows, i + half_size + 1))

        j_1 = np.max((0, j - half_size))
        j_1 = np.min((image_cols, j_1 + 1))
        j_2 = np.min((image_cols, j + half_size + 1))

        if i_1 == i_2 or j_1 == j_2:
            return None

        return image[int(i_1):int(i_2), int(j_1):int(j_2)]

# References:
#
# [1] Nunes, Cristiano FG, and Flávio LC Padua. "A Local Feature Descriptor
# Based on Log-Gabor Filters for Keypoint Matching in Multispectral Images."
# IEEE Geoscience and Remote Sensing Letters 14.10 (2017): 1850-1854.
#
# [2] Walia, Ekta, and Vishal Verma. "Boosting local texture descriptors with
# Log-Gabor filters response for improved image retrieval." International
# Journal of Multimedia Information Retrieval 5.3 (2016): 173-184.
