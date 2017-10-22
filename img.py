import cv2
import numpy as np
import matplotlib.pyplot as plt


def show(i, j):
    i = i.reshape((32, 32, 3))
    m, M = i.min(), i.max()
    cv2.imshow(str(j), (i - m) / (M - m))


def make(j, i):
    i = i.reshape((32, 32, 3))
    cv2.imwrite(j, i)


road = cv2.imread("road.jpg", cv2.IMREAD_COLOR)
bird = cv2.imread("bird.jpg", cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)


# # plt.show()
data = np.asarray([road, bird], dtype=np.float32)
data = data.reshape(-1, 3072)
show(data[1], 1)
center = data - np.mean(data, axis=1).reshape(-1, 1)
center /= np.std(data, axis=1).reshape(-1, 1)
show(center[1], 2)

# Assume input data matrix X of size [N x D]
# zero-center the data (important)
# get the data covariance matrix
cov = np.dot(center.T, center) / center.shape[0]
U, S, V = np.linalg.svd(cov, full_matrices=True)
Xrot = np.dot(center, U)  # decorrelate the data
Xrot_reduced = np.dot(center, U[:, :100])  # Xrot_reduced becomes [N x 100]
# whiten the data:
# divide by the eigenvalues (which are square roots of the singular values)
Xwhite = Xrot / np.sqrt(S + 1e-5)
show(Xwhite[1], 3)


# make('norm.jpg', data[1])
# make('center.jpg', center[1])
# make('diff.jpg', data[1] / 255 - center[1])

cv2.waitKey(0)
cv2.destroyAllWindows()


# cv2.imshow('color', color)
# cv2.imshow('gray', gray)
# cv2.imshow('mean', mean)
# cv2.imshow('gray', xrot_reduces)
