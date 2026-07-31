from abc import ABC, abstractmethod
import numpy as np

class BaseReconstructor(ABC):
    @abstractmethod
    def reconstruct(self, points: np.ndarray, normals: np.ndarray, **kwargs):
        """
        Args:
            points: NumPy array of shape (N, 3) representing point coordinates.
            normals: NumPy array of shape (N, 3) representing surface normals.

        Returns:
            vertices: NumPy array of shape (M, 3) representing reconstructed mesh vertices.
            faces: NumPy array of shape (F, 3) representing triangle face indices.
        """
        pass