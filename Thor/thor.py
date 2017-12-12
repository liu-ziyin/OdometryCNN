#!/usr/bin/env python

import numpy as np
from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage.io import imshow
import skimage as im
import random

import robosims

scaleFactor = 0.5
imgSize = int(300 * scaleFactor)
N_steps = 10

if __name__ == "__main__":

  import argparse
  import os

  parser = argparse.ArgumentParser()
  parser.add_argument("output_directory", type=str, default="out", nargs='?', help="Directory in which to generate data.")

  args = parser.parse_args()

  if not os.path.exists(args.output_directory):
    os.mkdir(args.output_directory)


  env = robosims.controller.ChallengeController(
    unity_path='thor-201705011400-Linux64',
    x_display="0.0", # this parameter is ignored on OSX, but you must set this to the appropriate display on Linux
    mode='continuous'
    ##unity_width= x_screen,
    ##unity_height= y_screen
    
  )
  env.start()

  angle = 0.0
  vertical = 0.0

  # intialize angles
  event = env.step(action=dict(action='Look', rotation=angle))
  event = env.step(action=dict(action='MoveLeft', moveMagnitude=20.0))


  print("Outputting data in %s/" % args.output_directory)
  print("Rendering %d steps." % N_steps)

  ang_vel = 0.0
  vertical_vel = 0.0
  angle_data = np.empty((N_steps, 2))
  
  image_data = np.empty((N_steps, imgSize, imgSize))

  for step in range(N_steps):
    #if step % 2 == 0:
    #ang_vel = np.random.uniform(low=-20, high=20)
    
    velList = [-20, 20]
    ang_vel = random.choice(velList)
    vertical_vel = np.random.uniform(low=-5, high=5)

    # Possible actions are: MoveLeft, MoveRight, MoveAhead, MoveBack, LookUp, LookDown, RotateRight, RotateLeft
    event = env.step(action=dict(action='Rotate', rotation=angle))
    event = env.step(action=dict(action='Look', horizon=vertical))
    event = env.step(action=dict(action='MoveAhead', moveMagnitude=1.0))
    angle_data[step] = (angle, vertical)

    image = rescale(event.frame, scaleFactor)
    image2 = rgb2gray(image)

    image_data[step] = np.fft.fftshift(image2)
    imshow(image)

    angle += ang_vel
    x = vertical + vertical_vel
    if (abs(x) < 10.0):
      vertical += vertical_vel

    if (step + 1) % 100 == 0:
      print("Stored %d/%d frames." % (step + 1, N_steps))

  angle_filename = os.path.join(args.output_directory, 'data_angles')
  image_filename = os.path.join(args.output_directory, 'data_images')
  print("Saving to %s.npy and %s.npy..." % (angle_filename, image_filename))
  np.save(angle_filename, angle_data)
  np.save(image_filename, image_data)
  print("Done.")
