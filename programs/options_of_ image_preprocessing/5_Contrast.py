import cv2
import numpy as np
from matplotlib import pyplot as plt

images_files = ['../images/image.jpg', '../images/only_stains.jpg', '../images/text.jpg']

space_width = 100
reference_image = cv2.imread(images_files[0])
height, width = reference_image.shape[:2]

masks = []
contours_images = []

for idx, img_file in enumerate(images_files):
    image = cv2.imread(img_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
    enhanced_gray = clahe.apply(gray)

    _, thresh = cv2.threshold(enhanced_gray, 150, 255, cv2.THRESH_BINARY_INV)

    mask_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    mask_resized = cv2.resize(mask_bgr, (width, height))
    masks.append(mask_resized)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    output_resized = cv2.resize(output, (width, height))
    contours_images.append(output_resized)

space = np.full((height, space_width, 3), (250, 200, 200),
                dtype=np.uint8)

final_mask_image = masks[0]
for mask in masks[1:]:
    final_mask_image = cv2.hconcat([final_mask_image, space, mask])

final_contours_image = contours_images[0]
for contours_image in contours_images[1:]:
    final_contours_image = cv2.hconcat([final_contours_image, space, contours_image])

cv2.imwrite('preprocessing/contrast_masks.jpg', final_mask_image)
cv2.imwrite('preprocessing/contrast_contours.jpg', final_contours_image)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.title("Combined Masks with Space")
plt.imshow(cv2.cvtColor(final_mask_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Contours on Original Images with Space")
plt.imshow(cv2.cvtColor(final_contours_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
