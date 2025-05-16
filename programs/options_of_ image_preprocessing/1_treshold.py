import cv2
import numpy as np

images_files = ['../images/image.jpg', '../images/only_stains.jpg', '../images/text.jpg']

images_files_output = [
    'THRESH_BINARY.jpg', 'contours_treshold_THRESH_BINARY.jpg',
    'THRESH_BINARY_INV.jpg', 'contours_treshold_THRESH_BINARY_INV.jpg',
    'THRESH_TRUNC.jpg', 'contours_treshold_THRESH_TRUNC.jpg'
]

types = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC]
space_width = 100

output_index = 0
for thresh_type in types:
    processed_images = []
    threshold_images = []

    for img_file in images_files:
        image = cv2.imread(img_file)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 5)

        _, thresh = cv2.threshold(denoised, 160, 255, thresh_type)
        threshold_images.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_with_contours = image.copy()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

        processed_images.append(image_with_contours)

    heights = [img.shape[0] for img in processed_images]
    max_height = max(heights)

    resized_images = [
        cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height))
        for img in processed_images
    ]
    resized_thresh_images = [
        cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height))
        for img in threshold_images
    ]

    space = np.full((max_height, space_width, 3), (255, 200, 200), dtype=np.uint8)

    final_thresh_with_spaces = []
    for i, img in enumerate(resized_thresh_images):
        final_thresh_with_spaces.append(img)
        if i < len(resized_thresh_images) - 1:
            final_thresh_with_spaces.append(space)

    final_thresh_image = cv2.hconcat(final_thresh_with_spaces)

    final_image_with_spaces = []
    for i, img in enumerate(resized_images):
        final_image_with_spaces.append(img)
        if i < len(resized_images) - 1:
            final_image_with_spaces.append(space)

    final_image = cv2.hconcat(final_image_with_spaces)

    cv2.imwrite(f"preprocessing/{images_files_output[output_index]}", final_thresh_image)
    cv2.imwrite(f"preprocessing/{images_files_output[output_index + 1]}", final_image)

    output_index += 2
