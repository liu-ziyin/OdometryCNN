import numpy as np

import engine

from utils import Camera, Line, Point, Polygon, save_image

N_poly = 100
N_steps = 500

# In pixels.
cam_size = 64
# In degrees.
cam_fov = 90

def random_poly():
  n_points = np.random.randint(3, 7)
  distance = np.random.uniform(2, 6)
  angle = np.random.uniform(0, 2 * np.pi)
  size = np.random.uniform(0.1, 0.8)

  loc = distance * np.array((np.cos(angle), np.sin(angle)))

  points = []
  for i in range(n_points):
    color = np.random.uniform(0.3, 1.0)
    dp = np.random.uniform(-1, 1, 2)
    points.append(Point(loc + size * dp, color))

  return Polygon(points)

if __name__ == "__main__":
  import argparse
  import os
  import time

  parser = argparse.ArgumentParser()
  parser.add_argument("output_directory", type=str, default="out", nargs='?', help="Directory in which to generate data.")

  args = parser.parse_args()

  print("Outputting data in %s/" % args.output_directory)
  print("Generating world with %d random polygons..." % N_poly)
  print("Rendering world at %d steps with a camera fov of %0.1f degrees and camera size of %d pixels" % (N_steps, cam_fov, cam_size))

  if not os.path.exists(args.output_directory):
    os.mkdir(args.output_directory)

  lines = []
  for i in range(N_poly):
    lines.extend(random_poly().lines)

  world = engine.render_world(lines, 75)
  save_image(os.path.join(args.output_directory, 'world.png'), world)

  times = []

  cam_pos = np.array((0, 0))
  cam_angle = 0
  max_angle_jump = np.radians(10)

  data = np.empty((N_steps, 3 + cam_size))
  for i in range(N_steps):
    cam = Camera(cam_pos, cam_angle, cam_fov, cam_size)

    start_time = time.time()
    img = engine.render(lines, cam)
    times.append(time.time() - start_time)

    data[i, 0:2] = cam_pos
    data[i, 2] = cam_angle
    data[i, 3:] = img

    save_image(os.path.join(args.output_directory, 'im_%03d.png' % i), img)

    cam_angle += np.random.uniform(-max_angle_jump, max_angle_jump)

  np.savetxt(os.path.join(args.output_directory, 'data.txt'), data, header="x, y, theta, pixels (%d)" % cam_size)
  print("Scene render time (s) (Avg. Max. Min.): %0.4f %0.4f %0.4f" % (sum(times) / len(times), max(times), min(times)))
