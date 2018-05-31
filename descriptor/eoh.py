import cv2
import numpy as np


class EOH():
    '''
    Python implementation by: Cristiano Fraga G. Nunes <cfgnunes@gmail.com>
    Original MATLAB source code: https://github.com/ngunsu/matlab-eoh-sift
    Described in the paper [1].

    This implementation may produces a different result
    from original MATLAB source code, because:
    - The float precision differ to locate the "z maximums";
    - The implementation of Canny detector is different.

    Extract the EOH (Edge Oriented Histogram).

    Input image should be a single-band image,
    but if it's a multiband (e.g. RGB) image
    only the 1st band will be used.

    The output descriptor is a array with 4x4x5=80 values.

    The image is split into 4x4 non-overlapping rectangular regions.
    For each region, 5 edge orientation histogram is computed,
    (horizontal, vertical, 2 diagonals and 1 non-directional).
    '''

    __NUMBER_ROW_REGIONS = 4
    __NUMBER_COL_REGIONS = 4

    def __init__(self, window_size=80):
        self.window_size = window_size

        # Initialize the filter bank
        self.__initialize_filter_bank()

    def descriptor_size(self):
        return int(self.__NUMBER_ROW_REGIONS *
                   self.__NUMBER_COL_REGIONS *
                   len(self.__FILTERS))

    def descriptor_type(self):
        return np.float32

    # Return a list of descriptors for each region around all keypoints.
    # The size of the region is defined by "window_size".
    def compute(self, image, keypoints):

        if len(keypoints) == 0:
            return

        descriptors = np.zeros((len(keypoints), self.descriptor_size()),
                               self.descriptor_type())

        for i, k in enumerate(keypoints):
            # Crop a region (window_size X window_size)
            # around the keypoint
            window_image = self.__crop_region(image,
                                              (k.pt[1], k.pt[0]),
                                              self.window_size)

            if window_image is None:
                continue

            maxp = self.__apply_filters(window_image)

            # Compute the descriptor for this region
            descriptors[i] = self.__compute_subregions(maxp)

        return keypoints, descriptors

    # Return the descriptor for a image
    def compute_descriptor(self, image):
        maxp = self.__apply_filters(image)
        descriptor = self.__compute_subregions(maxp)
        return descriptor

    # Apply the filters in the image and return a matrix containing
    # the index values of maximum filters responses.
    # Each value of this returned matrix represent which filter had a
    # maximum response for that pixel coordinate.
    def __apply_filters(self, image):
        if image.size == 0:
            return None

        if image.dtype != np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_rows, image_cols = image.shape
        image_float = image.copy().astype(np.float32)

        filtered_images = np.zeros(
            (image_rows, image_cols, self.__nfilters),
            np.float32)

        for i, f in enumerate(self.__FILTERS):
            filtered_images[:, :, i] = np.abs(
                cv2.filter2D(src=image_float,
                             ddepth=-1,
                             kernel=f,
                             dst=0,
                             anchor=(-1, -1),
                             delta=0,
                             borderType=cv2.BORDER_CONSTANT))

        # Matrix containing the index values of maximum filters responses.
        maxp = filtered_images.argmax(2)
        maxp += 1

        # Apply an Canny edge detector
        image_canny = self.__apply_canny(image)
        maxp = maxp * (image_canny > 0)

        return maxp

    # Compute the histogram for each subregion and return the descriptor.
    # Every pixel of each subregion contributes to a bin on
    # the histogram according to the filters.
    def __compute_subregions(self, maxp):
        descriptor = np.zeros(self.descriptor_size(), self.descriptor_type())
        image_rows, image_cols = maxp.shape

        step = 0
        for i in range(self.__NUMBER_ROW_REGIONS):
            for j in range(self.__NUMBER_COL_REGIONS):
                i1 = i * round(image_rows / self.__NUMBER_ROW_REGIONS)
                j1 = j * round(image_cols / self.__NUMBER_COL_REGIONS)
                i2 = (i + 1) * round(image_rows / self.__NUMBER_ROW_REGIONS)
                j2 = (j + 1) * round(image_cols / self.__NUMBER_COL_REGIONS)

                # Crop the subregion
                subregion = maxp[i1:i2, j1:j2]

                # Compute the histogram
                histogram = cv2.calcHist(
                    images=[subregion.astype(np.uint8)],
                    channels=None,
                    mask=None,
                    histSize=[self.__nfilters],
                    ranges=[1, self.__nfilters + 1])
                descriptor[step:(step + self.__nfilters)] = histogram.T
                step += self.__nfilters

        # Normalize the final descriptor
        cv2.normalize(descriptor, descriptor)
        return descriptor

    # Crop a region of (size X size) centered around a center reference
    def __crop_region(self, image, center, size):
        i, j = center
        image_rows, image_cols = image.shape

        i1 = np.max((0, i - size / 2))
        i1 = np.min((image_rows, i1))
        i2 = np.min((image_rows, i + size / 2))

        j1 = np.max((0, j - size / 2))
        j1 = np.min((image_cols, j1))
        j2 = np.min((image_cols, j + size / 2))

        if i1 == i2 or j1 == j2:
            return None

        return image[int(i1):int(i2), int(j1):int(j2)]

    def __apply_canny(self, image):
        # Apply an Gaussian filter to eliminate noise
        sigma = 3
        image_filtered = cv2.GaussianBlur(src=image,
                                          ksize=(sigma * 3, sigma * 3),
                                          sigmaX=sigma,
                                          sigmaY=sigma)

        # Find automatic good threshold values for Canny edge detector
        otsu, _ = cv2.threshold(src=image_filtered,
                                thresh=0,
                                maxval=255,
                                type=cv2.THRESH_OTSU)

        threshold = otsu * 0.5
        image_edge = cv2.Canny(image_filtered, 0.5 * threshold, threshold)

        return image_edge

    def __initialize_filter_bank(self):
        # Build a filter bank with Sobel filters
        f1 = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]  # Horizontal filter
        f2 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # Vertical filter
        f3 = [[2, 2, -1], [2, -1, -1], [-1, -1, -1]]  # 45 filter
        f4 = [[-1, 2, 2], [-1, -1, 2], [-1, -1, -1]]  # 135 filter
        f5 = [[-1, 0, 1], [0, 0, 0], [1, 0, -1]]  # No orientation filter

        self.__FILTERS = []
        self.__FILTERS.append(np.array(f1))
        self.__FILTERS.append(np.array(f2))
        self.__FILTERS.append(np.array(f3))
        self.__FILTERS.append(np.array(f4))
        self.__FILTERS.append(np.array(f5))

        self.__nfilters = len(self.__FILTERS)


'''
References:

[1] Aguilera, Cristhian, et al. "Multispectral image feature points."
Sensors 12.9 (2012): 12661-12672.
'''
