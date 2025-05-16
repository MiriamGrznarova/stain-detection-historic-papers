import cv2

def get_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        value = param['image'][y, x]
        print(f"{param['color_space']} values at position ({x}, {y}): {value}")

image = cv2.imread(fr"")
image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
color_spaces = {
    'HSV': [cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR],
    'LAB': [cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR],
    'YCrCb': [cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR]
}

for color_space, conversion in color_spaces.items():
    converted_image = cv2.cvtColor(image, conversion[0])
    window_name = f'{color_space} Image'
    cv2.imshow(window_name, cv2.cvtColor(converted_image,conversion[1]))
    cv2.setMouseCallback(window_name, get_value, param={'image': converted_image, 'color_space': color_space})

cv2.waitKey(0)
cv2.destroyAllWindows()