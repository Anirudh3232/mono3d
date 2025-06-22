from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
            self,
            level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)

        # Convert to numpy for processing
        level_np = level.detach().cpu().numpy()

        # Use PyMCubes for proper marching cubes
        try:
            import mcubes
            vertices, triangles = mcubes.marching_cubes(level_np, 0.0)
            v_pos = torch.from_numpy(vertices).float()
            t_pos_idx = torch.from_numpy(triangles.astype(np.int64)).long()
        except ImportError:
            # Fallback to scipy if mcubes is not available
            from scipy.spatial import Delaunay

            # Create a simple mesh using the level set
            x = np.linspace(0, 1, self.resolution)
            y = np.linspace(0, 1, self.resolution)
            z = np.linspace(0, 1, self.resolution)
            X, Y, Z = np.meshgrid(x, y, z)

            # Get points where level is close to 0
            mask = np.abs(level_np) < 0.1
            points = np.column_stack((X[mask], Y[mask], Z[mask]))

            # Create a simple surface
            if len(points) > 4:  # Need at least 4 points for a 3D surface
                tri = Delaunay(points)
                v_pos = torch.from_numpy(points).float()
                t_pos_idx = torch.from_numpy(tri.simplices).long()
            else:
                # Fallback to a simple cube if not enough points
                v_pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float)
                t_pos_idx = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=torch.long)

        return v_pos, t_pos_idx
