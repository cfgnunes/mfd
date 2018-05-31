import cv2
import numpy as np
import util.gaborfunctions as gf


class MFD():
    '''
    Extract the MFD (Multiespectral Feature Descriptor) descriptor

    Paper [1]:
       "A Local Feature Descriptor Based on Log-Gabor Filters
       for Keypoint Matching in Multispectral Images"
    Authors:
        Cristiano Fraga G. Nunes <cfgnunes@gmail.com>
        Flavio Luis Cardeal Padua <cardeal@decom.cefetmg.br>

    Input image should be a single-band image,
    but if it's a multiband (e.g. RGB) image
    only the 1st band will be used.

    The output descriptor is a array with 4x4x2x5=160 values.

    The image is split into 4x4 non-overlapping rectangular regions.
    For each region, 10 (2x5) Log-Gabor filter is computed.
    '''

    __NUMBER_ROW_REGIONS = 4
    __NUMBER_COL_REGIONS = 4

    def __init__(self, window_size=80, filter_scales=2, filter_orientations=5):
        self.window_size = window_size
        self.filter_scales = filter_scales
        self.filter_orientations = filter_orientations

    def descriptor_size(self):
        return int(self.__NUMBER_ROW_REGIONS *
                   self.__NUMBER_COL_REGIONS *
                   self.filter_scales * self.filter_orientations)

    def descriptor_type(self):
        return np.float32

    # Return a list of descriptors for each region around all keypoints.
    # The size of the region is defined by "window_size".
    def compute(self, image, keypoints):

        if len(keypoints) == 0:
            return

        maxp = self.__apply_filters(image)

        descriptors = np.zeros((len(keypoints), self.descriptor_size()),
                               self.descriptor_type())

        for i, k in enumerate(keypoints):
            # Crop a region (window_size X window_size)
            # around the keypoint
            window_maxp = self.__crop_region(maxp,
                                             (k.pt[1], k.pt[0]),
                                             self.window_size)
            if window_maxp is None:
                continue

            # Compute the descriptor for this region
            descriptors[i] = self.__compute_subregions(window_maxp)

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
        image_f = np.fft.fft2(image_float)

        self.__initialize_filter_bank((image_cols, image_rows))

        filtered_images = np.zeros((image_rows, image_cols, self.__nfilters),
                                   np.float32)

        for i, f in enumerate(self.__FILTERS):
            image_edge_f = image_f * np.fft.fftshift(f)
            image_back = np.fft.ifft2(image_edge_f)
            filtered_images[:, :, i] = np.abs(image_back)

        # Matrix containing the index values of maximum filters responses.
        maxp = filtered_images.argmax(2)
        maxp += 1

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

    def __initialize_filter_bank(self, ksize):
        # Build a filter bank with a Log-Gabor filters
        n_scales = self.filter_scales
        n_orient = self.filter_orientations

        # The values of the parameters for the log-Gabor
        # filters were defined as suggested in a previous
        # study [2], because they demonstrated good results
        # in texture extraction when log-Gabor filters were
        # used for image descriptions.
        min_wavelen = 3
        scale_factor = 2
        sigma_over_f = 0.65
        sigma_theta = 1

        self.__FILTERS = gf.get_log_gabor_filterbank(ksize,
                                                     n_scales,
                                                     n_orient,
                                                     min_wavelen,
                                                     scale_factor,
                                                     sigma_over_f,
                                                     sigma_theta)
        self.__nfilters = len(self.__FILTERS)


'''
References:

[1] Nunes, Cristiano FG, and Flavio LC Padua. "A Local Feature Descriptor
Based on Log-Gabor Filters for Keypoint Matching in Multispectral Images."
IEEE Geoscience and Remote Sensing Letters 14.10 (2017): 1850-1854.

[2] Walia, Ekta, and Vishal Verma. "Boosting local texture descriptors with
Log-Gabor filters response for improved image retrieval." International
Journal of Multimedia Information Retrieval 5.3 (2016): 173-184.
'''
