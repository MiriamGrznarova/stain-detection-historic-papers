import cv2
import numpy as np
from matplotlib import pyplot as plt

images_files = ['../images/image.jpg', '../images/only_stains.jpg', '../images/text.jpg']

first_image = cv2.imread(images_files[0])
height, width, _ = first_image.shape

processed_images = []
mask_images = []

space_width = 100

for img_file in images_files:
    image = cv2.imread(img_file)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lower_lab = np.array([140, 125, 130])
    upper_lab = np.array([165, 135, 135])
    mask = cv2.inRange(lab_image, lower_lab, upper_lab)

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = image.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    resized_mask = cv2.resize(masked_image, (width, height))
    resized_output = cv2.resize(output_image, (width, height))

    processed_images.append(resized_output)

    space = np.full((height, space_width, 3), (255, 200, 200), dtype=np.uint8)
    processed_images.append(space)

    mask_images.append(resized_mask)
    mask_images.append(space)

processed_images.pop()
mask_images.pop()

final_image_for_contours = cv2.hconcat(processed_images)

final_image_for_masks = cv2.hconcat(mask_images)

cv2.imwrite('preprocessing/LAB_contours.jpg', final_image_for_contours)
cv2.imwrite('preprocessing/LAB_masks.jpg', final_image_for_masks)

plt.figure(figsize=(15, 5))
plt.imshow(cv2.cvtColor(final_image_for_contours, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

plt.figure(figsize=(15, 5))
plt.imshow(cv2.cvtColor(final_image_for_masks, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
