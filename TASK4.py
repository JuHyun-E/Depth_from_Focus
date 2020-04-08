from pygco import cut_simple  # to apply graph-cut

import argparse
import cv2
import numpy as np
import os


# read image
def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return img


# save image
def save_image(save_path, save_as, img):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return cv2.imwrite(save_path + save_as, img)


# convert to gray scale
def cvt_to_grayscale(img):
    imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return imGray


def read_images_from_path(img_path):
    img_list = []
    for root, dirs, files in os.walk(img_path):
        for file in files:
            img_list.append(read_image(root + file))

    return img_list


def normalization(x):
    max_, min_ = np.max(x), np.min(x)
    normalized = (x - min_) / (max_ - min_)
    return normalized


# General
WEIGHTS = np.array(
        [[0, 0, 1, 2, 1, 0, 0],
         [0, 1, 2, 3, 2, 1, 0],
         [1, 2, 3, 4, 3, 2, 1],
         [2, 3, 4, 5, 4, 3, 2],
         [1, 2, 3, 4, 3, 2, 1],
         [0, 1, 2, 3, 2, 1, 0],
         [0, 0, 1, 2, 1, 0, 0]])


def graph_cut(img_list, gaussian_size, unary_scale, pair_scale, n_iter):
    imGray_list = []
    for img in img_list:
        imGray_list.append(cvt_to_grayscale(img))

    n = len(imGray_list)
    unary_cost = []
    ii, jj = np.meshgrid(range(n), range(n))
    pairwise_cost = np.abs(ii - jj) * pair_scale

    for imGray in imGray_list:
        gray_img = imGray.astype(np.float32) / 255.
        grad = np.exp(-(cv2.Sobel(gray_img, cv2.CV_32F, 1, 1) ** 2))
        unary_cost.append(cv2.GaussianBlur(grad, (gaussian_size, gaussian_size), 0) * unary_scale)

    unary_cost = normalization(np.stack(unary_cost, axis=-1)) * unary_scale
    graph_img = cut_simple(unary_cost.astype(np.int32), pairwise_cost.astype(np.int32), n_iter)

    return graph_img


def weighted_median_filter(img):
    wmf_img = np.zeros_like(img)

    height, width = img.shape
    kernel_size = len(WEIGHTS)
    MARGIN = int(kernel_size / 2)
    medIdx = int((np.sum(WEIGHTS) - 1) / 2)

    for i in range(MARGIN, height - MARGIN):
        for j in range(MARGIN, width - MARGIN):
            neighbors = []
            for k in range(-MARGIN, MARGIN + 1):
                for l in range(-MARGIN, MARGIN + 1):
                    a = img.item(i + k, j + l)
                    w = WEIGHTS[k + MARGIN, l + MARGIN]
                    for _ in range(w):
                        neighbors.append(a)
            neighbors.sort()
            median = neighbors[medIdx]
            wmf_img.itemset((i, j), median)

    return wmf_img


def main(basement_path):
    img_path = basement_path + "aligned/"
    graph_save_path = basement_path + "4.graph_cut/"
    weighted_median_filter_save_path = basement_path + "5.wmf/"
    save_as = "output.jpg"

    img_list = read_images_from_path(img_path)

    print("Obtaining graph-cut depth map ...")
    graph_img = graph_cut(img_list, gaussian_size=9, unary_scale=2 * 22, pair_scale=2 ** 12, n_iter=5)
    print("Saving graph-cut depth-map : ", graph_save_path)
    save_image(graph_save_path, save_as, graph_img)

    print("Obtaining weighted median filtered depth map ...")
    weighted_median_filter_img = weighted_median_filter(graph_img)
    print("Saving weighted median filtered depth map : ", weighted_median_filter_save_path)
    save_image(weighted_median_filter_save_path, save_as, weighted_median_filter_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-label optimization using graph-cuts and Depth Refinement(Weighed median filter)")
    parser.add_argument('--basement_path', default="./data/save/07/", help="After Aligning and Matching Images directory")
    args = parser.parse_args()

    basement_path = args.basement_path

    main(basement_path)
