import numpy as np
import math as m


class BeforeNeRF:

    def __init__(self, coordinates, k, r, pixel_coordinates):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.z = coordinates[2]
        self.theta, self.phi = self.compute_ray_direction(k, r, pixel_coordinates)

    def get_coordinates(self):
        return self.x, self.y, self.z

    def get_viewing_direction(self):
        return self.theta, self.phi

    def compute_ray_direction(self, k, r, pixel_coordinates):
        f_x, f_y = k[0, 0], k[1, 1]
        c_x, c_y = k[0, 2], k[1, 2]

        p_cam = np.array([(pixel_coordinates[0] - c_x) / f_x, (pixel_coordinates[1] - c_y) / f_y, 1])

        d_world = r.T @ p_cam

        d_world /= np.linalg.norm(d_world)

        return np.acos(d_world[2]), np.atan2(d_world[1], d_world[0])