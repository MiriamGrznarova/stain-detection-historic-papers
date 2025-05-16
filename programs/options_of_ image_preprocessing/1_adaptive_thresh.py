import cv2
import numpy as np
from matplotlib import pyplot as plt

images_files = ['../images/image.jpg', '../images/only_stains.jpg', '../images/text.jpg']
methods = {
    'mean_c': lambda img: cv2.adaptiveThreshold(img, 125, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 7),
    'gaussian_c': lambda img: cv2.adaptiveThreshold(img, 125, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7),
    'mean_c_blur': lambda img: cv2.adaptiveThreshold(cv2.medianBlur(img, 5), 125, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 7)
}

for method_name, method in methods.items():
    contours_images = []
    thresh_images = []

    for img_file in images_files:
        image = cv2.imread(img_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 5)

        thresh = method(denoised)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_with_contours = image.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

        contours_images.append(image_with_contours)
        thresh_images.append(thresh)

    heights = [img.shape[0] for img in contours_images]
    max_height = max(heights)

    resized_contours_images = [
        cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height))
        for img in contours_images
    ]
    resized_thresh_images = [
        cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height))
        for img in thresh_images
    ]

    resized_contours_images_rgb = [img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in resized_contours_images]
    resized_thresh_images_rgb = [img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in resized_thresh_images]

    space_width = 100
    space = np.full((max_height, space_width, 3), (255, 200, 200), dtype=np.uint8)

    final_thresh_with_spaces = []
    for i, img in enumerate(resized_thresh_images_rgb):
        final_thresh_with_spaces.append(img)
        if i < len(resized_thresh_images_rgb) - 1:
            final_thresh_with_spaces.append(space)

    final_thresh_image = cv2.hconcat(final_thresh_with_spaces)

    final_contours_with_spaces = []
    for i, img in enumerate(resized_contours_images_rgb):
        final_contours_with_spaces.append(img)
        if i < len(resized_contours_images_rgb) - 1:
            final_contours_with_spaces.append(space)

    final_contours_image = cv2.hconcat(final_contours_with_spaces)

    method_name_clean = method_name.replace("_", "-")
    cv2.imwrite(f'preprocessing/{method_name_clean}_contours.jpg', final_contours_image)
    cv2.imwrite(f'preprocessing/{method_name_clean}_thresh.jpg', final_thresh_image)

plt.figure(figsize=(15, 5))

for method_name in methods.keys():
    contours_img = cv2.imread(f'preprocessing/{method_name.replace("_", "-")}_contours.jpg')
    thresh_img = cv2.imread(f'preprocessing/{method_name.replace("_", "-")}_thresh.jpg')

    plt.subplot(3, len(methods), list(methods.keys()).index(method_name) + 1)
    plt.title(f"{method_name} Contours")
    plt.imshow(cv2.cvtColor(contours_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, len(methods), len(methods) + list(methods.keys()).index(method_name) + 1)
    plt.title(f"{method_name} Threshold")
    plt.imshow(cv2.cvtColor(thresh_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

plt.tight_layout()
plt.show()
