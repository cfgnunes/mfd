'''
MFD (Multiespectral Feature Descriptor)

Python implementation by the author: Cristiano Nunes <cfgnunes@gmail.com>
See the paper [1].
'''

import numpy as np
import phasepack


class MFD():
    '''
    Compute the MFD (Multiespectral Feature Descriptor).
    '''

    _N_SUB_REGIONS = 4  # Number of SxS subregions

    def __init__(self, window_size=80, n_scales=2, n_orient=5):
        self._window_size = window_size
        self._n_scales = n_scales
        self._n_orient = n_orient
        self._n_filters = n_scales * n_orient

    def compute(self, image, keypoints):
        """Compute the descriptor for each keypoint.

        Args:
            image (ndarray): input image.
            keypoints (ndarray): keypoints.

        Returns:
            list of ndarray: descriptors for each keypoint.
        """

        if not keypoints:
            return None

        descriptors = np.zeros(
            (len(keypoints), self.descriptor_size()), self.descriptor_type())

        # Apply the Log-Gabor filters in the input image
        image_responses = self._apply_log_gabor_filters(image)

        # Compute the 'Maximum Index Map'
        mim = self._compute_maximum_index_map(np.vstack(image_responses))

        # Compute the descriptor for each keypoint
        for i, keypoint in enumerate(keypoints):

            # Crop a window image around the keypoint
            mim_window = self._crop_window(
                mim, (keypoint.pt[1], keypoint.pt[0]), self._window_size)

            if mim_window is None:
                continue

            # Compute the histogram for all subregions
            histograms_window = self._compute_histogram_subregions(
                mim_window, self._n_filters, self._N_SUB_REGIONS)

            # Concatenate all histograms in a single array
            histogram = np.concatenate(histograms_window)

            # Normalize the histogram to unit length
            histogram = histogram / np.linalg.norm(histogram)

            descriptors[i] = histogram

        return keypoints, np.array(descriptors)

    def descriptor_size(self):
        """Get the size of the descriptor.

        Returns:
            int: the size of the descriptor.
        """
        return int(self._N_SUB_REGIONS * self._N_SUB_REGIONS * self._n_filters)

    @staticmethod
    def descriptor_type():
        """Get the type of the descriptor.

        Returns:
            float: the type of the descriptor.
        """
        return np.float32

    def _apply_log_gabor_filters(self, image):
        """Apply the log-Gabor filters in the input image.

        Args:
            image (ndarray): input image.

        Returns:
            ndarray: image responses to Log-Gabor filters.
                     Shape: (Scales X Orientations X Rows X Cols).
        """

        # The values of the parameters for the log-Gabor filters were defined as
        # suggested in a previous study [2], because they demonstrated good
        # results in texture extraction when Log-Gabor filters were used for
        # image description.
        min_wavelen = 3
        scale_factor = 2
        sigma_on_f = 0.65

        # Build a filter bank with Log-Gabor filters
        _, _, _, _, _, image_responses, _ = phasepack.phasecong(
            image, nscale=self._n_scales, norient=self._n_orient,
            minWaveLength=min_wavelen, mult=scale_factor,
            sigmaOnf=sigma_on_f, k=2.0, g=3)

        # Compute the magnitude (absolute values) of the image responses
        image_responses = np.abs(image_responses)

        # Convert list of lists in a single array
        image_responses_array = np.swapaxes(image_responses, 0, 1)

        return image_responses_array

    @staticmethod
    def _compute_maximum_index_map(images):
        """Compute the 'Maximum Index Map': matrix containing the index values
        of maximum filters responses. Each value of this matrix represent which
        filter had a maximum response for that pixel coordinate.

        Args:
            images (ndarray): a list of images.
                              Shape: (Filters X Rows X Cols).

        Returns:
            ndarray: matrix containing the index values of maximum filters
                     responses (Maximum Index Map).
                     Shape: (Rows X Cols).
        """

        # Compute the argmax
        maximum_index_map = np.argmax(images, axis=0)
        maximum_index_map += 1

        return maximum_index_map

    @staticmethod
    def _compute_histogram_subregions(image, n_bins, n_subregions):
        """Split a image into subregions and compute its histograms.

        Args:
            image (ndarray): input image.
            n_bins (int): number of bins in the histogram.
            n_subregions (int): number of SxS subregions.

        Returns:
            list of ndarray: list of histogram for each subregion.
        """

        histograms = []
        image_rows, image_cols = image.shape

        # Note: splitting the window into subregions keeps significant spatial
        # information in the final descriptor.
        for i in range(n_subregions):
            for j in range(n_subregions):
                i_1 = i * round(image_rows / n_subregions)
                j_1 = j * round(image_cols / n_subregions)
                i_2 = (i + 1) * round(image_rows / n_subregions)
                j_2 = (j + 1) * round(image_cols / n_subregions)

                # Crop a subregion from the 'Maximum Index Map'
                subregion = image[i_1:i_2, j_1:j_2]

                # Compute the histogram for the subregion
                histogram, _ = np.histogram(subregion.astype(np.uint8),
                                            bins=n_bins, range=[1, n_bins + 1])

                # Add the histogram of the subregion to the histogram list
                histograms.append(histogram)

        return histograms

    @staticmethod
    def _crop_window(image, center, size):
        """Crop a squared window from a image.

        Args:
            image (ndarray): input image.
            center ((int, int)): coordinates of the window's center.
            size (int): squared size of the window.

        Returns:
            ndarray: output window image.
        """

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
# [1] C. F. G. Nunes and F. L. C. Pádua, "A local feature descriptor based on
# log-gabor filters for keypoint matching in multispectral images," IEEE
# Geoscience and Remote Sensing Letters, vol. 14, no. 10, pp. 1850-1854, Oct.
# 2017, DOI: 10.1109/lgrs.2017.2738632.
#
# [2] E. Walia and V. Verma, "Boosting local texture descriptors with log-gabor
# filters response for improved image retrieval," International Journal of
# Multimedia Information Retrieval, vol. 5, no. 3, pp. 173–184, apr 2016, DOI:
# 10.1007/s13735-016-0099-2.
