import numpy as np
import cv2


ROTATE_LIMIT = 0.15
SCALE_LIMIT = 0.15
SHIFT_LIMIT = 0.15


def _random_hue_saturation_value(image, hue_shift_limit=(-180, 180),
                                 sat_shift_limit=(-255, 255),
                                 val_shift_limit=(-255, 255), u=0.5):
    # Note! This function works with three channel images only!
    if np.random.random() < u:
        channels = cv2.split(image)
        channels = channels[3:]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        if (len(channels) > 0):
            rgb = cv2.split(image)
            image = cv2.merge(tuple(rgb + channels))

    return image


def _random_shift_scale_rotate(image,
                               shift_limit=(-0.0625, 0.0625),
                               scale_limit=(-0.1, 0.1),
                               rotate_limit=(-45, 45), aspect_limit=(0, 0),
                               borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(
            image,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=(0, 0, 0,)
        )

    return image


def _random_horizonal_flip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)

    return image


def perturb(img):
    if img is None:
        raise ArgumentException('Image is None')
    img = _random_hue_saturation_value(img,
                                       hue_shift_limit=(-50, 50),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))

    img = _random_shift_scale_rotate(img,
                                     shift_limit=(-1. * SHIFT_LIMIT, SHIFT_LIMIT),
                                     scale_limit=(-1. * SCALE_LIMIT, SCALE_LIMIT),
                                     rotate_limit=(-1. * ROTATE_LIMIT, ROTATE_LIMIT))
    img = _random_horizonal_flip(img)
    return img