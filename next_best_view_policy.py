import numpy as np
from sklearn.cluster import KMeans
from typing import Callable, Dict, Tuple

class RegionStrategy:
    """
    Base class for region partitioning strategies.
    """
    def compute_regions(self, camera_poses: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute regions based on camera poses.
        """
        raise NotImplementedError("This method should be implemented by subclasses!")

class KMeansRegionStrategy(RegionStrategy):
    def __init__(self, num_regions: int):
        """
        Initialize KMeans region strategy.
        """
        self.num_regions = num_regions

    def compute_regions(self, camera_poses: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute regions using KMeans clustering.
        """
        kmeans = KMeans(n_clusters=self.num_regions, random_state=42)
        labels = kmeans.fit_predict(camera_poses)
        return {i: camera_poses[labels == i] for i in range(self.num_regions)}

class SphericalRegionStrategy(RegionStrategy):
    def __init__(self, num_azimuths: int, num_elevations: int):
        """
        Initialize Spherical region strategy.
        """
        self.num_azimuths = num_azimuths
        self.num_elevations = num_elevations

    def compute_regions(self, camera_poses: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute regions based on spherical coordinates.
        """
        x, y, z = camera_poses[:, 0], camera_poses[:, 1], camera_poses[:, 2]
        azimuths = np.arctan2(y, x)
        elevations = np.arccos(z / np.linalg.norm(camera_poses, axis=1))

        azimuth_bins = np.linspace(-np.pi, np.pi, self.num_azimuths + 1)
        elevation_bins = np.linspace(0, np.pi / 2, self.num_elevations + 1)

        azimuth_indices = np.digitize(azimuths, azimuth_bins) - 1
        elevation_indices = np.digitize(elevations, elevation_bins) - 1

        region_labels = azimuth_indices + elevation_indices * self.num_azimuths

        regions = {}
        for label in np.unique(region_labels):
            regions[label] = camera_poses[region_labels == label]

        return regions

class HemisphereViewSelector:
    def __init__(
        self,
        radius: float = 1.0,
        num_circles: int = 5,
        poses_per_circle: int = 30,
        region_strategy: RegionStrategy = None,
        entropy_function: Callable[[np.ndarray], float] = None,
    ):
        """
        Initialize Hemisphere View Selector.
        
        :param radius: Maximum radius of the hemisphere.
        :param num_circles: Number of horizontal circles on the hemisphere.
        :param poses_per_circle: Number of candidate poses per circle.
        :param region_strategy: Instance of a region partitioning strategy.
        :param entropy_function: Custom function to calculate entropy.
        """
        self.radius = radius
        self.num_circles = num_circles
        self.poses_per_circle = poses_per_circle
        self.camera_poses = self._generate_candidate_poses()
        self.region_strategy = region_strategy or KMeansRegionStrategy(num_regions=10)
        self.entropy_function = entropy_function or self._default_entropy_function

    def _generate_candidate_poses(self) -> np.ndarray:
        """
        Generate candidate camera poses on the hemisphere.
        """
        poses = []
        for i in range(self.num_circles):
            circle_radius = self.radius * (i + 1) / self.num_circles
            theta = np.arccos(1 - 2 * (i + 1) / (self.num_circles + 1))
            phi = np.linspace(0, 2 * np.pi, self.poses_per_circle, endpoint=False)
            x = circle_radius * np.sin(theta) * np.cos(phi)
            y = circle_radius * np.sin(theta) * np.sin(phi)
            z = circle_radius * np.cos(theta)
            poses.extend(np.vstack((x, y, z)).T)
        return np.array(poses)

    def _default_entropy_function(self, view_points: np.ndarray) -> float:
        """
        Default entropy calculation function.
        """
        return np.mean(view_points)

    def compute_regions(self) -> Dict[int, np.ndarray]:
        """
        Compute regions based on the specified region partitioning strategy.
        """
        return self.region_strategy.compute_regions(self.camera_poses)

    def select_next_best_view(self) -> Tuple[np.ndarray, float]:
        """
        Select the next best view with the highest average entropy within each region.
        """
        regions = self.compute_regions()
        max_entropy = -1
        best_view = None
        for region_id, points in regions.items():
            entropy = self.entropy_function(points)
            if entropy > max_entropy:
                max_entropy = entropy
                best_view = points[np.random.choice(len(points))]
        return best_view, max_entropy