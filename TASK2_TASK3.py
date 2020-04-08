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


def focus_stack(aligned_img, gaussian_size, laplacian_size):
    imGray = cvt_to_grayscale(aligned_img)
    gaussian_img = cv2.GaussianBlur(imGray, (gaussian_size, gaussian_size), 0)
    laplacian_img = cv2.Laplacian(gaussian_img, cv2.CV_64F, ksize=laplacian_size)

    return laplacian_img


# High response of sharp edges
# 2. Focus Measure
def focus_measure_calculation(cost_volume, kernel_size=9):
    focus_measure = np.zeros_like(cost_volume)
    kernel = np.ones((kernel_size, kernel_size))

    for i in range(len(cost_volume)):
        focus_img = cost_volume[i]
        focus_measure[i] = focus_img * focus_img
        focus_measure[i] = cv2.filter2D(focus_measure[i], -1, kernel)

    return focus_measure


def all_in_focus(img_list, cost_volume, kernel_size, gaussian_size):
    bgr_imgs = np.asarray(img_list)  # BGR(blue,green,red)

    all_in_focus_img = np.zeros_like(bgr_imgs[0])
    height, width, channels = all_in_focus_img.shape

    focus_measure = focus_measure_calculation(cost_volume, kernel_size)
    argmax = np.argmax(focus_measure, axis=0)

    normalized = 255 - (normalization(argmax) * 255)
    depth_map = cv2.GaussianBlur(normalized, (gaussian_size, gaussian_size), 0)

    for i in range(height):
        for j in range(width):
            idx = argmax[i, j]
            all_in_focus_img[i, j, :] = bgr_imgs[idx, i, j, :]

    return depth_map, all_in_focus_img


def main(basement_path):
    img_path = basement_path + "aligned/"
    focus_save_path = basement_path + "1.focus_stack/"
    depth_save_path = basement_path + "2.depth_map/"
    all_focus_save_path = basement_path + "3.all_in_focus/"
    save_as = "output.jpg"

    # load aligned images
    img_list = read_images_from_path(img_path)
    stacked_focus_imgs = []

    print("Stacking focus using LoG ... ")
    for i, aligned_img in enumerate(img_list):
        focus_save_as = "focus_" + str(i) + ".jpg"
        # 1. Blur Estimation at Edges
        laplacian_img = focus_stack(aligned_img, gaussian_size=5, laplacian_size=5)
        stacked_focus_imgs.append(laplacian_img)

        print("... Saving images {} ...".format(str(i)))
        save_image(focus_save_path, focus_save_as, laplacian_img)

    # 3. Cost Volumne (Make 3D matrix)
    cost_volume = np.asarray(stacked_focus_imgs)

    print("Extracting focus from each images ...")
    depth_map, all_in_focus_img = all_in_focus(img_list, cost_volume, kernel_size=64, gaussian_size=5)  # Change appropriately kernel_size
    print("Saving initial_depth_from_focus image : ", depth_save_path)
    save_image(depth_save_path, save_as, depth_map)
    print("Saving all_in_focus image : ", all_focus_save_path)
    save_image(all_focus_save_path, save_as, all_in_focus_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initial depth from focus measure and All-in-focus image")
    parser.add_argument('--basement_path', default="./data/save/07/", help="After Aligning and Matching Images directory")
    args = parser.parse_args()

    basement_path = args.basement_path

    main(basement_path)
