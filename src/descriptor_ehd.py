import cv2
import numpy as np


class EHD():
    '''
    Python implementation by: Cristiano Nunes <cfgnunes@gmail.com>
    Original MATLAB source code: https://github.com/ngunsu/matlab-eoh-sift
    Described in the papers [1] and [2].

    This implementation may produces a different result
    from original MATLAB source code, because:
    - The float precision differ to locate the "z maximums".

    Extract the MPEG-7 EHD (Edge Orientation Histogram) descriptor.

    Input image should be a single-band image,
    but if it's a multiband (e.g. RGB) image
    only the 1st band will be used.

    The output descriptor is a array with 4x4x5=80 values.

    The image is split into 4x4 non-overlapping rectangular regions.
    For each region, 5 edge orientation histogram is computed,
    (horizontal, vertical, 2 diagonals and 1 non-directional).
    '''

    _N_ROW_REGIONS = 4
    _N_COL_REGIONS = 4

    def __init__(self, window_size=80):
        self.window_size = window_size

        # Initialize the filter bank
        self._initialize_filter_bank()

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

        descriptors = np.zeros(
            (len(keypoints), self.descriptor_size()), self.descriptor_type())

        for i, k in enumerate(keypoints):
            # Crop a region (window_size X window_size)
            # around the keypoint
            window_image = self._crop_region(
                image, (k.pt[1], k.pt[0]), self.window_size)

            if window_image is None:
                continue

            maxp = self._apply_filters(window_image)

            # Compute the descriptor for this region
            descriptors[i] = self._compute_subregions(maxp)

        return keypoints, descriptors

    # Return the descriptor for a image
    def compute_descriptor(self, image):
        maxp = self._apply_filters(image)
        descriptor = self._compute_subregions(maxp)
        return descriptor

    # Apply the filters in the image and return a matrix containing
    # the index values of maximum filters responses.
    # Each value of this returned matrix represent which filter had a
    # maximum response for that pixel coordinate.
    def _apply_filters(self, image):
        if image.size == 0:
            return None

        if image.dtype != np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_rows, image_cols = image.shape
        image_float = image.copy().astype(np.float32)

        filtered_images = np.zeros((image_rows, image_cols, self._nfilters),
                                   np.float32)

        for i, fil in enumerate(self._filters):
            filtered_images[:, :, i] = np.abs(
                cv2.filter2D(src=image_float, ddepth=-1, kernel=fil))

        # Matrix containing the index values of maximum filters responses.
        maxp = filtered_images.argmax(2)
        maxp += 1

        return maxp

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

    def _initialize_filter_bank(self):
        # Build a filter bank with Sobel filters
        fil1 = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]  # Horizontal filter
        fil2 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # Vertical filter
        fil3 = [[2, 2, -1], [2, -1, -1], [-1, -1, -1]]  # 45 filter
        fil4 = [[-1, 2, 2], [-1, -1, 2], [-1, -1, -1]]  # 135 filter
        fil5 = [[-1, 0, 1], [0, 0, 0], [1, 0, -1]]  # No orientation filter

        self._filters = []
        self._filters.append(np.array(fil1))
        self._filters.append(np.array(fil2))
        self._filters.append(np.array(fil3))
        self._filters.append(np.array(fil4))
        self._filters.append(np.array(fil5))

        self._nfilters = len(self._filters)

# References:
#
# [1] Wu, Peng, et al. "Texture descriptors in MPEG-7." International
# Conference on Computer Analysis of Images and Patterns. Springer,
# Berlin, Heidelberg, 2001.
#
# [2] Won, Chee Sun, Dong Kwon Park, and Soo‐Jun Park. "Efficient Use
# of MPEG‐7 Edge Histogram Descriptor." ETRI journal 24.1 (2002): 23-30.
