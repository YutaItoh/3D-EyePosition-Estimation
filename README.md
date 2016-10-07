
***NOTE: A new 3D eye tracking code is available on*** [***my new project***](https://github.com/YutaItoh/3D-Eye-Tracker/blob/master/README.md)

3D-EyePosition-Estimation
=========================

An eye-tracking software designed for  color, close-up eye images

This program computes an eyeball position in 3D given a calibrated camera image.

## Why eye(-pose) tracker?
With the growing market of Virtual-/Augmented-Reality (VR/AR) headsets such as Oculus Rift, Google Glass, and Microsoft HoloLens, the eye-tracking technology has become not only a key user interface, but also the prime technique to realize more realistic, emmersive VR/AR experieces.

Since I am working on this field, and needed to develop an eye-position tracking software for HMD calibration, 
I decided to share my preliminary, yet working, code with you.

A particular feature of the current implementation is that it estimates the center position (and 2-DoF orientation) of an eyeball in 3D.

## What it does:
Given a closeup image of an eye, this program estimates 
the 3D potision of the center and the oriation (two rotation axes) of an eye ball:
![](https://cloud.githubusercontent.com/assets/7195124/5328538/902631b8-7d83-11e4-95bf-192203a4115c.png)
(Left) A detected limbus (iris) area, (Middle left) 2D iris ellipse candidates from extracted edge segments, (Middle right) Best-fit 2D iris ellipse, (Right) Final 3D eyeball pose(s) estimated from the 2D ellipse.

## Overview
 main.cpp should give you a brief usage.
 In short, there are two steps to detect eyeball position from an image:
 1. eye_tracker::IrisDetector::DetectIris detects iris region and outputs a 2D iris ellipse (eye_tracker::Ellipse), then
 2. the ellipse computes the eyeball position given a few eyeball parameters (eye_tracker::Ellipse::ComputeEllipsePose).
 
 eye_tracker::IrisDetector is in iris_detector.h and eye_tracker::Ellipse in iris_geometry.h

 By default, some debug output are enabled by preprocessor macros (DEBUG_*) in iris_detector.cpp and iris_geometry.h.

## How to compile:
A CMakeLists.txt is provided for building a solution via CMake.
 
Platform:
 - Only tested on Windows (7, x86 and x64)
 
Requirements:
 - OpenCV 2.4.X (not 3.0.0 aplha yet)
 - Eigen 3
 - Boost > 1.49
 - Intel TBB

You would need to set their root directories manually in Cmake, search their library names to find related variables in Cmake GUI when you get an error.

 In addition to the above libraries, this repository contains codes from the following external open-source libraries:
 - A pupil tracker by <a href="http://www.cl.cam.ac.uk/research/rainbow/projects/pupiltracking/">Lech Swirski</a>
 - LSD (from OpenCV 3.0.0a source): a Line Segment Detector by <a href="http://www.ipol.im/pub/art/2012/gjmr-lsd/">Rafael Grompone von Gioi</a>. We use an OpenCV-3.0.0a version (BSD license)
 in ./external

The former is slightly modified (PupilTracker.cpp) and the latter is editted so that it works on OpenCV 2.X.
 
## How to use:
 Run main.exe in ./bin_win_x86 (vc10, x86 build) to get the example result above.
```cmd
>> main.exe [image file]
```

## Limitations
- Debug build crashes with an error related to TBB used in the iris detector.
- The code is not optimized, thus not real-time capable, yet
- The tracking is unstable for light-color eyes

## Note
- The algorithm returns *two* possible eye positions, so we need a postprocessing to determine the right one. Our work in the reference introduces a disambiguation method
- Our coodinate system is *right-handed* and the image origin is assumed to be *top-left*. In other words, we follow the OpenGL convention, not OpenCV.


## Reference:
Please refer to the following publication, which explaines the detail of the algorithm with a code table:
```latex
@article{itoh2014-3dui
  author    = {Itoh, Yuta and Klinker, Gudrun},
  title     = {Interaction-Free Calibration for Optical See-Through 
               Head-Mounted Displays based on 3D Eye Localization},
  booktitle = {{Proceedings of the 9th IEEE Symposium on 3D User Interfaces (3D UI)}},
  month     = {march},
  pages     = {75-82},
  year      = {2014}
}
```

## Licence
This repository is provided under MIT license.
