import numpy as np

from utils import in_cone, intersect_line_cone

def render(lines, camera):
  depth_buffer = -np.ones(camera.size)
  img = np.zeros(camera.size)

  for line in lines:
    start, end = line

    start_cam = camera.world_to_cam(start.point)
    end_cam = camera.world_to_cam(end.point)

    f_u = camera.f_u
    f_d = camera.f_d

    start_in = in_cone(start_cam, f_u, f_d)
    end_in = in_cone(end_cam, f_u, f_d)

    start_color = start.color
    end_color = end.color

    ### Line in frustrum detection ###
    if not start_in or not end_in:
      ts_u, ts_d = intersect_line_cone((start_cam, end_cam), (f_u, f_d))

      if not ts_u and not ts_d:
        assert not start_in and not end_in
        continue

      if not start_in:
        if not ts_u:
          ts, cone_side = ts_d, f_d
        elif not ts_d:
          ts, cone_side = ts_u, f_u
        else:
          assert ts_d and ts_u
          if ts_d[0] < ts_u[0]:
            ts, cone_side = ts_d, f_d
          else:
            ts, cone_side = ts_u, f_u

        start_cam = ts[1] * cone_side
        start_color = start.color + ts[0] * (end.color - start.color)

      if not end_in:
        if not ts_u:
          ts, cone_side = ts_d, f_d
        elif not ts_d:
          ts, cone_side = ts_u, f_u
        else:
          assert ts_d and ts_u
          if ts_d[0] > ts_u[0]:
            ts, cone_side = ts_d, f_d
          else:
            ts, cone_side = ts_u, f_u

        end_cam = ts[1] * cone_side
        end_color = start.color + ts[0] * (end.color - start.color)
    ### Line in frustrum detection ###

    # Camera point collision... is this safe to discard?
    if start_cam[0] < 1e-15 or end_cam[0] < 1e-15:
      print("Camera point collision")
      continue

    start_y = -start_cam[1] / start_cam[0]
    end_y = -end_cam[1] / end_cam[0]

    start_depth = start_cam[0]
    end_depth = end_cam[0]

    start_ind = int(round((camera.size / 2) + (camera.size / 2) * (start_y / camera.max_y)))
    end_ind = int(round((camera.size / 2) + (camera.size / 2) * (end_y / camera.max_y)))

    if start_ind > end_ind:
      start_ind, end_ind = end_ind, start_ind
      start_color, end_color = end_color, start_color
      stast_depth, end_depth = end_depth, start_depth

    if start_ind != end_ind:
      color_diff_per_pixel = (end_color - start_color) / (end_ind - start_ind)
      depth_per_pixel = (end_depth - start_depth) / (end_ind - start_ind)

    color = start_color
    depth = start_depth
    for ind in range(start_ind, end_ind):
      assert depth > 0

      if depth_buffer[ind] < 0 or depth < depth_buffer[ind]:
        depth_buffer[ind] = depth
        img[ind] = color

      color += color_diff_per_pixel
      depth += depth_per_pixel

  return img

def render_world(lines, pixels_per_unit):
  PIXEL_BORDER = 25

  all_points = np.empty((len(lines) * 2, 2))
  for i, line in enumerate(lines):
    all_points[2 * i, :] = line.start.point
    all_points[2 * i + 1, :] = line.end.point

  min_x, min_y = np.min(all_points, axis=0)
  max_x, max_y = np.max(all_points, axis=0)

  width = int(round((max_x - min_x) * pixels_per_unit + 2 * PIXEL_BORDER))
  height = int(round((max_y - min_y) * pixels_per_unit + 2 * PIXEL_BORDER))

  origin = np.array((min_x, max_y))

  def world2pixel(p):
    p_img = p - origin
    # Positive y is negative row index in the image.
    p_img[0], p_img[1] = -p_img[1], p_img[0]
    return np.round(p_img * pixels_per_unit) + np.array((PIXEL_BORDER, PIXEL_BORDER))

  img = np.zeros((height, width))

  for line in lines:
    start = world2pixel(line.start.point)
    end = world2pixel(line.end.point)
    # Manhattan distance.
    pixels = int(np.sum(np.abs(end - start)))

    if pixels:
      ds = (end - start) / pixels
      dcolor = (line.end.color - line.start.color) / pixels

    inds = start
    color = line.start.color
    for i in range(pixels):
      int_inds = np.round(inds).astype(int)
      img[int_inds[0], int_inds[1]] = color

      inds += ds
      color += dcolor

  return img
