#!/usr/bin/env python

import numpy as np 
try:
    import cv2
except:
    import sys
    sys.path.remove(sys.path[1])
    import cv2
import math
import argparse
import glob
from scipy.optimize import least_squares

def findHomo(imgList, wc):
    all_corners = []
    all_cr=[]
    H = []
    for img in imgList:
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(imgGray,(9,6),None)
        tempCorn =[]
        for i in range(54):
            xy = np.array([corners[i][0][0], corners[i][0][1]], dtype= np.float64)
            tempCorn.append(xy)
        tempCorn = np.array(tempCorn).reshape(54,2)
        all_corners.append(tempCorn)
        all_cr.append(corners)
        # bottom left
        c1 = tuple([corners[0][0][0], corners[0][0][1]])
        #top left
        c2 = tuple([corners[8][0][0], corners[8][0][1]])
        #top right
        c3 = tuple([corners[53][0][0], corners[53][0][1]])
        #tbottom right
        c4 = tuple([corners[45][0][0], corners[45][0][1]])
        dst = np.array([c1,c2,c3,c4], dtype = np.float32)
        tempH_w2i = cv2.getPerspectiveTransform(wc, dst)
        H.append(tempH_w2i)
    return H, all_corners, all_cr

def vij(h,i,j):
    v1 = h[0][i]*h[0][j]
    v2 = h[0][i]*h[1][j] + h[1][i]*h[0][j]
    v3 = h[1][i]*h[1][j]
    v4 = h[2][i]*h[0][j] + h[0][i]*h[2][j]
    v5 = h[2][i]*h[1][j] + h[1][i]*h[2][j]
    v6 = h[2][i]*h[2][j]
    
    v = np.array([v1,v2,v3,v4,v5,v6], dtype=np.float64).reshape(-1,1)
    return v

def VMat(allH):
    finalV = []
    for h in allH:
        v12 = vij(h,0,1).T
        finalV.append(v12)
        v11 = vij(h,0,0)
        v22 = vij(h,1,1)
        diff = (v11 -v22).T
        finalV.append(diff)
    V_arr = (np.array(finalV)).reshape(-1,6)
    _, _, Vh = np.linalg.svd(V_arr)
    b = Vh[5:] # shape = (1,6)
    b = b[0] # shape = (6,)
    return V_arr, b

def find_K(b):
    b11, b12, b22, b13, b23, b33 = b[0], b[1], b[2], b[3], b[4], b[5]
    v0 = (b12*b13 - b11*b23)/(b11*b22 - b12**2)
    lbda = b33 - ((b13**2 + v0*(b12*b13 - b11*b23))/b11) 
    alpha = np.sqrt(lbda/b11)
    beta = np.sqrt(lbda*b11/(b11*b22 - b12**2))
    gamma = -b12*(alpha**2)*beta/lbda
    u0 = (gamma*v0/beta) - (b13*(alpha**2)/lbda)
    K = np.array([[alpha, gamma, u0],[0.0, beta, v0],[0.0, 0.0,1.0]])
    return K

def getRT(allH, K):
    allRT =[]
    Kinv = np.linalg.inv(K)
    for i in range(len(allH)):
        t_homo = allH[i]
        Ainv_H = np.matmul(Kinv,t_homo)
        denum = np.linalg.norm(Ainv_H[:,0])
        lbda = 1./denum
        s= Ainv_H*lbda
        r1 = s[:,0].reshape(3,1)
        r2 = s[:,1].reshape(3,1)
        r3 = np.cross(r1.T, r2.T).reshape(3,1)
        t1 = s[:,2].reshape(3,1)
        tempRT = np.hstack((r1,r2,r3,t1))
        allRT.append(tempRT)
    return allRT

def fun(x, allCorn, allW, allH):
    # param = np.array([A[0,0],A[0,1],A[0,2],A[1,1],A[1,2],0.0,0.0])
    A = np.zeros([3,3], dtype= np.float64)
    A[0,0] = x[0]
    A[0,1] = x[1]
    A[0,2] = x[2]
    A[1,1] = x[3]
    A[1,2] = x[4]
    A[2,2] = 1
    k1, k2 = x[5], x[6]
    allRT = getRT(allH, A)
    error = []
    for n in range(len(allH)):
        RT = allRT[n] # size of allW is (54,4)
        camPts = np.matmul(RT, allW.T) # (x,y,z) known ideal world points in image(camera) cordinate i.e normalized points
        camNorm = camPts/camPts[2] # (x',y',1) homogenous form size (3,54)
        camNorm = camNorm.T # sie is (54,3)
        camSqrSum = camNorm[:,0]**2 + camNorm[:,1]**2 # size (54,1)
        distort = k1*camSqrSum + k2*(camSqrSum**2) # size (54,1)
        u0 = A[0,2]
        v0 = A[1,2]
        m = allCorn[n] # ideal observed image pixels corners, size of m is (54,2)
        mhat = np.matmul(np.matmul(A,RT),allW.T) # real estimated image pixels corners
        mhat = (mhat/mhat[2]).T # (u',v',1) as (3,54)
        ucap = (mhat[:,0] + (mhat[:,0] - u0)*distort).reshape(54,1)
        vcap = (mhat[:,1] + (mhat[:,1] - v0)*distort).reshape(54,1)
        mcap = np.hstack([ucap, vcap]) # distorted observed image points, size (54, 2) i.e (u, v) fir each point
        diff = (m - mcap) # reprojection error (3,54)
        err  = np.sum(np.linalg.norm(diff, axis =1)**2) # diff_u**2 + diff_v**2
        error.append(err)
    return np.array(error)

def main():
    files = glob.glob('./Calibration_Imgs/*.jpg')
    # given, reference block size is 21.5 units
    # note XY refer to horizontal and vertical of image (not to the axis marked on the reference image)
    wCordXY = np.array([[21.5, 21.5],
                        [21.5,21.5*9],
                        [21.5*6, 21.5*9],
                        [21.5*6,21.5]], dtype= np.float32)

    allImgs= []
    allImgOrg = []
    for File in files:
        img = cv2.imread(File)
        imgOrg =img.copy()
        allImgOrg.append(imgOrg)
        allImgs.append(img)

    allH, all_corn, allcr = findHomo(allImgs, wCordXY)
    vmat, b = VMat(allH)
    A = find_K(b)
    # allRT = getRT(allH, A)
    K = [0,0]
    print('Initial estimate for Camera Intrinsic Matrix -')
    print(A)
    print('')
    param = np.array([A[0,0],0.0,A[0,2],A[1,1],A[1,2],0.0,0.0])
    allW_xy = []
    for i in range(6):
        for j in range(9):
            allW_xy.append(np.array([21.5*(i+1), 21.5*(j+1),0,1]))
    allW_xy = np.array(allW_xy, dtype =np.float64).reshape(54,4)

    res = least_squares(fun, x0=param, method='lm', args=(all_corn, allW_xy, allH))
    print('Minimization convergance status= ', res.success)
    print('')
    A_opt = np.zeros([3,3], dtype= np.float)
    A_opt[0,0] = res.x[0]
    A_opt[0,1] = res.x[1]
    A_opt[0,2] = res.x[2]
    A_opt[1,1] = res.x[3]
    A_opt[1,2] = res.x[4]
    A_opt[2,2] = 1
    k_opt = [res.x[5], res.x[6]]
    print('Optimised Distortion Coefficients -')
    print(k_opt)
    print('')
    print('Optimized Camera Intrinsic Matrix -')
    print(A_opt)
    print('')
    dist_opt = np.array([k_opt[0],k_opt[1],0,0,0])
    allRT_opt = getRT(allH, A_opt)
    err= []
    allW_xy1 = np.array(allW_xy[:,:3], dtype =np.float32)

    for i in xrange(len(allH)):
        R = allRT_opt[i][:,0:3]
        rvec,_ = cv2.Rodrigues(R)
        t = allRT_opt[i][:,3]
        imgpoints2, _ = cv2.projectPoints(allW_xy1, rvec, t, A_opt,dist_opt)
        diff = allcr[i] - imgpoints2
        error = np.sum(np.linalg.norm(diff,axis=1))/len(imgpoints2)
        err.append(error)
    err =np.array(err)
    rejErr = np.mean(err)
    print('Re-projection error -')
    print(rejErr)
    print('')
    print('Generating undistorted images')
    count =1
    for image in allImgOrg:
        undistImg = cv2.undistort(image,A_opt,dist_opt)
        cv2.imwrite('./undist_output/'+str(count)+'_undistort.png',undistImg)
        count+=1

if __name__ == '__main__':
    main()