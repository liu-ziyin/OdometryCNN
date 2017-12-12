from collections import namedtuple

import numpy as np
import scipy.misc

Point = namedtuple('Point', ['point', 'color'])
Line = namedtuple('Line', ['start', 'end'])

class Polygon:
  def __init__(self, points):
    self.lines = []
    for i in range(len(points)):
      end_ind = i + 1
      if end_ind == len(points):
        end_ind = 0

      self.lines.append(Line(points[i], points[end_ind]))

class Camera:
  def __init__(self, pos, angle, fov, size):
    '''
      pos: Position of the camera.
      angle: Angle of the camera in radians.
      fov: Field of view of the camera in degrees.
      size: Number of pixels in the camera.
    '''
    self.pos = pos
    self.cam2world = np.array(((np.cos(angle), -np.sin(angle)),
                               (np.sin(angle), np.cos(angle))))
    self.fov = np.radians(fov)
    self.size = size

    # abs(y_value) of the farthest point in view after
    # projection to plane with x = 1 in the camera frame.
    self.max_y = np.tan(self.fov / 2)

    # Positive y frustrum cone side
    self.f_u = np.array((np.cos(self.fov / 2), np.sin(self.fov / 2)))
    # Negative y frustrum cone side
    self.f_d = self.f_u.copy()
    self.f_d[1] = -self.f_d[1]

  def world_to_cam(self, point):
    return self.cam2world.T.dot(point - self.pos)

def intersect_line_cone(line, cone_dirs):
  # Returns t, s such that intersection = s * cone_direction = line[0] + t * (line[1] - line[0])
  start_cam, end_cam = line
  ret = []

  A = np.column_stack((end_cam - start_cam, -cone_dirs[0]))
  b = -start_cam

  for cone_direction in cone_dirs:
    A[:, 1] = -cone_direction
    try:
      t, s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
      ret.append(None)
      continue

    if t < 0 or t > 1 or s < 0:
      ret.append(None)
    else:
      ret.append((t, s))

  return ret

def cross(x, y):
  return x[0] * y[1] - x[1] * y[0]

def in_cone(point, f_u, f_d):
  return cross(f_u, point) < 0 and cross(f_d, point) > 0

def save_image(fname, img):
  img_8bit = np.empty(np.shape(img), dtype=np.uint8)
  img_8bit[...] = 255 * img

  output_shape = np.shape(img)
  if img.ndim == 1:
    output_shape = (1, output_shape[0])

  scipy.misc.imsave(fname, np.reshape(img_8bit, output_shape))
