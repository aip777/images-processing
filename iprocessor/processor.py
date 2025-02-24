import numpy as np
import tifffile as tiff
from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA

class ImageProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.image = tiff.imread(file_path)
        self.metadata = self.extract_metadata()

    def extract_metadata(self):
        return {
            "shape": self.image.shape,
            "dtype": str(self.image.dtype)
        }

    def get_slice(self, z: int = None, time: int = None, channel: int = None):
        sliced_image = self.image  # Keep original image shape
        print(sliced_image)
        if self.image.ndim != 5:
            raise ValueError(f"Expected a 5D image, but got {self.image.ndim}D")

        if z is not None:
            sliced_image = sliced_image[:, :, z, :, :]  # (100, 100, 5, 3)

        if time is not None:
            sliced_image = sliced_image[:, :, :, time, :]  # (100, 100, 3)

        if channel is not None:
            sliced_image = sliced_image[:, :, :, :, channel]  # (100, 100)

        return sliced_image

    def compute_statistics(self):
        return {
            "mean": np.mean(self.image, axis=(0, 1, 2)),
            "std": np.std(self.image, axis=(0, 1, 2)),
            "min": np.min(self.image, axis=(0, 1, 2)),
            "max": np.max(self.image, axis=(0, 1, 2)),
        }

    def perform_pca(self, n_components=3):
        reshaped = self.image.reshape(-1, self.image.shape[-1])
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(reshaped)
        return reduced.reshape(*self.image.shape[:-1], n_components)

    def segment_image(self, channel=0):
        channel_data = self.image[..., channel]
        threshold = threshold_otsu(channel_data)
        return (channel_data > threshold).astype(np.uint8)
