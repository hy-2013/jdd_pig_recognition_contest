#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
description: 
author: clzhang
date: 06/12/2017
"""

import os
import cv2

frames_per_image = 5
skip_num_per_valid = 15

base_path = "JDD_contest/sample" + str(frames_per_image)
train_path = base_path + "/train/"
valid_path = base_path + "/valid/"
models_path = base_path + "/models/"

"""生成目录"""
os.system('mkdir ' + base_path + ' ' + train_path + ' ' + valid_path + ' ' + models_path)
os.system('cd ' + train_path + '&& ' + 'mkdir pig17 pig18 pig19 pig20 pig21 pig22 pig23 pig24 pig25 pig26 pig27 pig28 pig29 pig30 pig1 pig2 pig3 pig4 pig5 pig6 pig7 pig8 pig9 pig10 pig11 pig12 pig13 pig14 pig15 pig16')
os.system('cd ' + valid_path + '&& ' + 'mkdir pig17 pig18 pig19 pig20 pig21 pig22 pig23 pig24 pig25 pig26 pig27 pig28 pig29 pig30 pig1 pig2 pig3 pig4 pig5 pig6 pig7 pig8 pig9 pig10 pig11 pig12 pig13 pig14 pig15 pig16')

# video to image
for idx in range(1, 31):
    vidcap = cv2.VideoCapture('JDD_contest/train_video/%d.mp4' % idx)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      if count % frames_per_image == 0 and success:
          print('Read a new frame: ', idx, count, success)
          cv2.imwrite(train_path + "pig%d/pig%d_%d.jpg" % (idx, idx, count), image)  # save frame as JPEG file
      count += 1


# move some image from train_dir to valid_dir
for idx in range(1, 31):
    for count in range(0, 2950):
        if count % skip_num_per_valid == 0:
            print('Read a new frame: ', idx, count)
            mv_command = "mv " + train_path + "pig%d/pig%d_%d.jpg " % (idx, idx, count) +  valid_path + "pig%d" % idx
            os.system(mv_command)
        count += 1