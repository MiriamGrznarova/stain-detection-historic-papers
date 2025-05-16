import cv2
import numpy as np
from matplotlib import pyplot as plt

images_files = ['../images/image.jpg', '../images/only_stains.jpg', '../images/text.jpg']

reference_image = cv2.imread(images_files[0])
height, width = reference_image.shape[:2]

masks = []
contours_images = []

for idx, img_file in enumerate(images_files):
    image = cv2.imread(img_file)
    original_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((4, 4), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    mask_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    mask_resized = cv2.resize(mask_bgr, (width, height))
    masks.append(mask_resized)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_contours = original_image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

    contours_resized = cv2.resize(image_with_contours, (width, height))
    contours_images.append(contours_resized)

space_width = 100
max_height = max([img.shape[0] for img in masks])
space = np.full((max_height, space_width, 3), (255, 200, 200), dtype=np.uint8)

final_image_with_spaces = []
for i, img in enumerate(masks):
    final_image_with_spaces.append(img)
    if i < len(masks) - 1:
        final_image_with_spaces.append(space)

final_mask_image = cv2.hconcat(final_image_with_spaces)

final_thresh_with_spaces = []
for i, img in enumerate(contours_images):
    final_thresh_with_spaces.append(img)
    if i < len(contours_images) - 1:
        final_thresh_with_spaces.append(space)

final_contours_image = cv2.hconcat(final_thresh_with_spaces)

cv2.imwrite('preprocessing/morph_masks.jpg', final_mask_image)
cv2.imwrite('preprocessing/morph_contours.jpg', final_contours_image)
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.title("Combined Masks")
plt.imshow(cv2.cvtColor(final_mask_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Contours on Original Images")
plt.imshow(cv2.cvtColor(final_contours_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
