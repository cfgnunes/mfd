"""
MFD (Multiespectral Feature Descriptor).

Implementation by: Cristiano Fraga G. Nunes <cfgnunes@gmail.com>

References:
[1] https://dx.doi.org/10.1109/lgrs.2017.2738632
Nunes, Cristiano F. G., and Flávio L. C. Pádua. "A local feature descriptor
based on log-Gabor filters for keypoint matching in multispectral images." IEEE
Geoscience and Remote Sensing Letters 14.10 (2017): 1850-1854.
"""

import numpy as np
import phasepack


class MFD():
    """MFD (Multiespectral Feature Descriptor)."""

    # A constant to split the region into NxN subregions
    _N_SUBREGIONS = 4

    def __init__(self, window_size=80, n_scales=2, n_orient=5):
        """Class constructor.

        Args:
            window_size (int, optional):
                image window size around the keypoint, image patch.
            n_scales (int, optional): number of scale Log-Gabor filters.
            n_orient (int, optional): number of orientation Log-Gabor Filters.
        """
        self._window_size = window_size
        self._n_scales = n_scales
        self._n_orient = n_orient
        self._n_filters = n_scales * n_orient

    def compute(self, image, keypoints):
        """Compute the descriptor for each keypoint.

        Args:
            image (numpy.ndarray): input image.
            keypoints (numpy.ndarray): keypoints.

        Returns:
            numpy.ndarray: descriptors for each keypoint.
                Shape: (n_keypoints, descriptor_size).
        """
        if not keypoints:
            return None

        descriptors = np.zeros(
            (len(keypoints), self.descriptor_size()), self.descriptor_type())

        # Compute the 'Maximum Index Map'
        mim = self._compute_maximum_index_map(image)

        # Compute the descriptor for each keypoint
        for i, keypoint in enumerate(keypoints):

            # Crop a window image around the keypoint
            mim_window = self._crop_window(
                mim, keypoint.pt, self._window_size)

            if mim_window is None:
                continue

            # Compute the histogram for all subregions
            histograms_window = self._compute_histogram_subregions(
                mim_window, self._n_filters, self._N_SUBREGIONS)

            # Concatenate all histograms in a single array
            histogram = np.concatenate(histograms_window)

            # Normalize the histogram to unit length
            norm = np.linalg.norm(histogram)
            if norm > 0:
                histogram = histogram / norm

            descriptors[i] = histogram

        return keypoints, descriptors

    def _compute_maximum_index_map(self, image):
        """Compute the 'Maximum Index Map' from a input image.

        Compute a 'Maximum Index Map': a matrix containing the index
        values of maximum filters responses. Each matrix value represents
        which filter had a maximum response for that pixel coordinate.

        Args:
            image (numpy.ndarray): the input image.

        Returns:
            numpy.ndarray: a matrix containing the index values of maximum
                filters responses (Maximum Index Map). Shape: (height, width).
        """
        # Apply the Log-Gabor filters in the input image
        # Parameters from https://dx.doi.org/10.1007/s13735-016-0099-2
        image_responses = self._apply_log_gabor_filters(
            image,
            n_scales=self._n_scales,
            n_orient=self._n_orient,
            min_wavelen=3,
            scale_factor=2,
            sigma_on_f=0.65)

        # Compute the 'Maximum Index Map' using argmax
        mim = np.argmax(np.vstack(image_responses), axis=0)

        return mim

    def descriptor_size(self):
        """Get the size of the descriptor.

        Returns:
            int: the size of the descriptor.
        """
        return self._N_SUBREGIONS * self._N_SUBREGIONS * self._n_filters

    @staticmethod
    def descriptor_type():
        """Get the type of the descriptor.

        Returns:
            numpy.float32: the type of the descriptor.
        """
        return np.float32

    @staticmethod
    def _apply_log_gabor_filters(image, n_scales, n_orient,
                                 min_wavelen, scale_factor, sigma_on_f):
        """Apply the log-Gabor filters in the input image.

        Args:
            image (numpy.ndarray): input image.
            n_scales (int, optional): scale filters.
            n_orient (int, optional): orientation filters.
            min_wavelen (float): Log-Gabor parameter min_wavelen.
            scale_factor (float): Log-Gabor parameter scale_factor.
            sigma_on_f (float): Log-Gabor parameter sigma_on_f.

        Returns:
            numpy.ndarray: image responses to Log-Gabor filters.
                Shape: (n_scales, n_orientations, height, width).
        """
        # Build a filter bank with Log-Gabor filters
        _, _, _, _, _, image_responses, _ = phasepack.phasecong(
            image, nscale=n_scales, norient=n_orient,
            minWaveLength=min_wavelen, mult=scale_factor,
            sigmaOnf=sigma_on_f, k=1.0, g=3)

        # Compute the magnitude (absolute values) of the image responses
        image_responses = np.abs(image_responses)

        # Convert a list of lists into a single array
        image_responses = np.swapaxes(image_responses, 0, 1)

        return image_responses

    @staticmethod
    def _compute_histogram_subregions(image, n_bins, n_subregions):
        """Split an image into subregions and compute its histograms.

        Args:
            image (numpy.ndarray): input image.
            n_bins (int): number of bins in the histogram.
            n_subregions (int): number of NxN subregions.

        Returns:
            list of numpy.ndarray: list of histograms for each subregion.
        """
        histograms = []
        image_h, image_w = image.shape[:2]

        # Note: splitting the window into subregions keeps important spatial
        # information in the final descriptor.
        for i in range(n_subregions):
            for j in range(n_subregions):
                y_1 = i * int(image_h / n_subregions + 0.5)
                x_1 = j * int(image_w / n_subregions + 0.5)
                y_2 = (i + 1) * int(image_h / n_subregions + 0.5)
                x_2 = (j + 1) * int(image_w / n_subregions + 0.5)

                # For non-squared regions (if the splitting is not exact)
                if i == n_subregions - 1:
                    y_2 += image_h - y_2
                if j == n_subregions - 1:
                    x_2 += image_w - x_2

                # Crop a subregion from the 'Maximum Index Map'
                subregion = image[y_1:y_2, x_1:x_2]

                # Compute the histogram for the subregion
                histogram, _ = np.histogram(
                    subregion, bins=n_bins, range=[0, n_bins - 1])

                # Add the histogram of the subregion to the histogram list
                histograms.append(histogram)

        return histograms

    @staticmethod
    def _crop_window(image, center, size):
        """Crop a squared window from an image.

        Args:
            image (numpy.ndarray): input image.
            center (tuple of (int, int)): coord. of the center (x, y).
            size (int): the squared size of the window.

        Returns:
            numpy.ndarray: output window image.
        """
        image_h, image_w = image.shape[:2]
        center_x, center_y = int(center[0] + 0.5), int(center[1] + 0.5)
        half_size = size // 2
        size_even = 1 - size % 2

        # Check the boundaries
        y_1 = max(0, center_y - half_size)
        x_1 = max(0, center_x - half_size)
        y_2 = min(image_h, center_y + half_size + 1 - size_even)
        x_2 = min(image_w, center_x + half_size + 1 - size_even)

        # Ignore incomplete windows
        if y_2 - y_1 != size or x_2 - x_1 != size:
            return None

        return image[y_1:y_2, x_1:x_2]
