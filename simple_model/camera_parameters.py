import numpy as np

class CameraParameters:
    def __init__(self):
        # Intrinsic parameters
        self.fx = None  # Focal length in x direction
        self.fy = None  # Focal length in y direction
        self.cx = None  # Principal point x-coordinate
        self.cy = None  # Principal point y-coordinate
        self.K = None   # Intrinsic matrix

        # Distortion parameters
        self.k1 = None  # Radial distortion coefficient 1
        self.k2 = None  # Radial distortion coefficient 2
        self.p1 = None  # Tangential distortion coefficient 1
        self.p2 = None  # Tangential distortion coefficient 2
        self.k3 = None  # Radial distortion coefficient 3

        # Extrinsic parameters
        self.R = None   # Rotation matrix
        self.t = None   # Translation vector

    def set_intrinsics(self, fx, fy, cx, cy):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])

    def set_distortion(self, k1, k2, p1, p2, k3=0): # We follow the OpenCV convention
        self.k1, self.k2, self.p1, self.p2, self.k3 = k1, k2, p1, p2, k3

    def set_extrinsics(self, R, t):
        self.R, self.t = R, t

    def to_vector(self):
        return np.array([self.fx, self.fy, self.cx, self.cy, 
                         self.k1, self.k2, self.p1, self.p2, self.k3])

    @classmethod
    def from_vector(cls, vector):
        params = cls()
        params.set_intrinsics(vector[0], vector[1], vector[2], vector[3])
        params.set_distortion(vector[4], vector[5], vector[6], vector[7], vector[8])
        return params
