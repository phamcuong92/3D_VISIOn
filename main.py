from PIL import Image
from numpy import *
from pylab import *
import numpy as np

from utils import camera, sfm, homography, sift

# Read features
im1 = array(Image.open('./image/test_image1.jpg'))
sift.process_image('./image/test_image1.jpg', 'im1.sift')
l1, d1 = sift.read_features_from_file('im1.sift')

im2 = array(Image.open('./image/test_image2.jpg'))
sift.process_image('./image/test_image2.jpg', 'im2.sift')
l2, d2 = sift.read_features_from_file('im2.sift')

matches = sift.match_twosided(d1, d2)
print(matches.shape)

ndx = matches.nonzero()[0]
print(ndx)
x1 = homography.make_homog(l1[ndx, :2].T)
print(l1[ndx, :2].shape)
ndx2 = [int(matches[i]) for i in ndx]
x2 = homography.make_homog(l2[ndx2, :2].T)

x1n = x1.copy()
x2n = x2.copy()
print(len(ndx))
figure(figsize=(16,16))
sift.plot_matches(im1, im2, l1, l2, matches, True)
show()


def F_from_ransac(x1, x2, model, maxiter=5000, match_threshold=1e-6):
    """ Robust estimation of a fundamental matrix F from point
    correspondences using RANSAC (ransac.py from
    http://www.scipy.org/Cookbook/RANSAC).

    input: x1, x2 (3*n arrays) points in hom. coordinates. """

    from utils import ransac
    data = np.vstack((x1, x2))
    d = 20 # 20 is the original
    # compute F and return with inlier index
    F, ransac_data = ransac.ransac(data.T, model,
                                   8, maxiter, match_threshold, d, return_all=True)
    return F, ransac_data['inliers']
#
# find F through RANSAC
model = sfm.RansacModel()
F, inliers = F_from_ransac(x1n, x2n, model, maxiter=5000, match_threshold=1e-6)
#
# print(len(x1n[0]))
# print(len(inliers))
# Camera projection
P1 = array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2 = sfm.compute_P_from_fundamental(F)
#
# triangulate inliers and remove points not in front of both cameras
X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2)
# # plot the projection of X
cam1 = camera.Camera(P1)
cam2 = camera.Camera(P2)
x1p = cam1.project(X)
x2p = cam2.project(X)
figure()
imshow(im1)
gray()
plot(x1p[1], x1p[0], 'o')
plot(x1[1], x1[0], 'r.')
axis('off')

figure()
imshow(im2)
gray()
plot(x2p[1], x2p[0], 'o')
plot(x2[1], x2[0], 'r.')
axis('off')
show()

figure(figsize=(16, 16))
im3 = sift.appendimages(im1, im2)
im3 = vstack((im3, im3))

imshow(im3)

cols1 = im1.shape[1]
rows1 = im1.shape[0]
for i in range(len(x1p[0])):
    if (0<= x1p[0][i]<cols1) and (0<= x2p[0][i]<cols1) and (0<=x1p[1][i]<rows1) and (0<=x2p[1][i]<rows1):
        plot([x1p[0][i], x2p[0][i]+cols1],[x1p[1][i], x2p[1][i]],'c')
axis('off')
show()

# 3D plot
from mpl_toolkits.mplot3d import axes3d
fig = figure()
ax = fig.gca(projection='3d')
ax.plot(-X[0],X[1],X[2],'k.')
axis('off')

show()