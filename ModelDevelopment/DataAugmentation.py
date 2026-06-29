import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import random


# region class

class Image_Augmentation:

    # region Methods ...

    # Function _4

    def save_augmented_images(self,output_folder="AugmentedDataSet"):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for folder in folders:
            class_dir = os.path.join(output_folder, folder, folder)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            class_images = final_images[folder]
            number_of_images = len(class_images)

            for id, img in enumerate(class_images):
                filename = f"{folder}_{id}.png"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, img)

        for folder in folders:
            class_dir = os.path.join(output_folder, folder, folder)
            if os.path.exists(class_dir):
                number_of_images = len(os.listdir(class_dir))
                print(f"number of images in {folder}: {number_of_images}")

    # Function _3
    def flip_image(self,img, flip_code):
        return cv2.flip(img, flip_code)

    # Function _2
    def rotate_image(self,img, angle):
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        return rotated_img

    # Function _1
    def augment_image_advanced(self,img, angles=None):
        augmented_images = []
        angle_info = []

        if angles is None:
            angles = [-45, -30, -15, 0, 15, 30, 45]

        for angle in angles:
            rotated = self.rotate_image(img, angle)
            augmented_images.append(rotated)
            angle_info.append(f"rotate_{angle}")

        flipped_h = self.flip_image(img, 1)
        augmented_images.append(flipped_h)
        angle_info.append(f"flip_horizontal")

        flipped_v = self.flip_image(img, 0)
        augmented_images.append(flipped_v)
        angle_info.append(f"flip_vertical")

        # flip & rotate
        for angle in [-20, 20, 30]:
            rotated = self.rotate_image(img, angle)
            flipped_rotated = self.flip_image(rotated, 1)
            augmented_images.append(flipped_rotated)
            angle_info.append(f"rotate_{angle}_flip")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, 2] = hsv[:, 2] * np.random.uniform(0.6, 1.4)
        bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        augmented_images.append(bright)
        angle_info.append(f"brightness")

        # noise
        noise = np.random.normal(0, 0.05, img.shape) * 255
        noisy = img + noise

        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        augmented_images.append(noisy)
        angle_info.append(f"noise")

        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        random_angle = random.choice([-25, -10, 10, 25])
        blurred_rotated = self.rotate_image(blurred, random_angle)
        augmented_images.append(blurred_rotated)
        angle_info.append(f"blur_rotate_{random_angle}")

        return augmented_images, angle_info


# endregion

path = '..\\OriginalDataSet\\*'

folders = []
for folder in glob.glob(path):
    folders.append(folder.split('\\')[-1])

# print(folders)

count_of_images = 2500

final_images = dict()
final_labels = dict()

base_path = 'OriginalDataSet'

for folder in folders:
    folder_path = os.path.join(base_path, folder, folder)
    image_files = glob.glob(os.path.join(folder_path, "*"))

    # print(image_files)
    original_count = len(image_files)
    print(original_count)

    all_images = []
    all_angles = []
    original_images = []

    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            original_images.append(img)
            all_images.append(img)
            all_angles.append("original_0")

    if len(all_images) < count_of_images:
        needed_count = count_of_images - len(all_images)
        print(f"needed {needed_count} images")

        for id, image in enumerate(original_images):

            image_augmentation_object = Image_Augmentation()

            aug_imgs, aug_angles = image_augmentation_object.augment_image_advanced(image)

            for aug_img, aug_angle in zip(aug_imgs, aug_angles):
                if len(all_images) < count_of_images:
                    all_images.append(aug_img)
                    all_angles.append(aug_angle)


    final_images[folder] = all_images[:count_of_images]
    final_labels[folder] = [folder] * count_of_images

# print(f'final images: {final_images}')
# print(f'final labels: {final_labels}')

all_final_images = []
all_final_labels = []

for folder in folders:
    all_final_images.extend(final_images[folder])
    all_final_labels.extend(final_labels[folder])
#
# print(f'all final images: {all_final_images}')
# print(f'all final labels: {all_final_labels}')

image_augmentation_object = Image_Augmentation()
image_augmentation_object.save_augmented_images("..\\AugmentedDataSet")