# Camera Calibration

## Overview
Estimating parameters of the camera like the focal length, distortion coefficients and principle point is called Camera Calibration. It is one of the most time consuming and important part of any computer vision research involving 3D geometry. An automatic way to perform efficient and robust camera calibration would be wonderful. One such method was presented Zhengyou Zhang of Microsoft in this paper and is regarded as one of the hallmark papers in camera calibration.<br>

<p align="center">
	<img src="https://github.com/varunasthana92/Camera_Auto_Calibation/blob/master/Calibration_Imgs/IMG_20170209_042612.jpg" width="300">
	<img src="https://github.com/varunasthana92/Camera_Auto_Calibation/blob/master/Calibration_Imgs/IMG_20170209_042630.jpg" width="300">
</p>

We are trying to get a good initial estimate of the parameters so that we can feed it into the non-linear optimizer. Non-linear Geometric Error Minimization done using least_squares function from the scipy library.

## Dependencies
* python v2.7.17
* numpy v1.16.6 
* scipy v0.19.1
* OpenCV v3.2.0

## Input data
Input image is of a 10 X 7 chessboard. Thirteen such images taken from a Google Pixel XL phone with a locked focus is provided (at random pose). For calibration, it is a general practise to se the inner grid (here 9 X 6).<br>
Actual size of each square is 21.5mm.

## How to run
```
$ python wrapper.py
```