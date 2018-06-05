import random
import math
import numpy as np
import cv2
import skimage


# geometric ---
def pad_to_factor(image, factor=16):
    """
    padding image to make 16 being its factor
    :param image:
    :param factor:
    :return:
    """
    height, width = image.shape[:2]
    h = math.ceil(height / factor) * factor
    w = math.ceil(width / factor) * factor

    image = cv2.copyMakeBorder(image,
                               top=0,
                               bottom=h - height,
                               left=0, right=w - width,
                               borderType=cv2.BORDER_REFLECT101,
                               value=[0, 0, 0])
    return image


def pad_to_size(image, mask, width, height):
    """
    padding image to specified size
    :param image:
    :return:
    """
    img_height, img_width = image.shape[:2]
    assert height > img_height
    assert width > img_width
    delta_w = width - img_width
    delta_h = height - img_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                               borderType=cv2.BORDER_CONSTANT,
                               value=[0, 0, 0])
    mask  = cv2.copyMakeBorder(mask, top, bottom, left, right,
                               borderType=cv2.BORDER_CONSTANT,
                               value=[0, 0, 0])
    return image, mask


def resize_to_factor(image, mask, factor=16):
    H, W = image.shape[:2]
    h = (H // factor) * factor
    w = (W // factor) * factor
    return fix_resize_transform(image, mask, w, h)


def fix_resize_transform(image, mask, w, h):
    H, W = image.shape[:2]
    if (H, W) != (h, w):
        image = cv2.resize(image, (w, h))

        mask = mask.astype(np.float32)
        mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)
        mask = mask.astype(np.int32)
    return image, mask


def fix_crop_transform(image, mask, x, y, w, h):
    """
    (x,   y)--------(x+w,   y)
      |                 |
      |                 |
    (x, y+h)--------(x+w, y+h)
    """

    H, W = image.shape[:2]
    assert (H >= h)
    assert (W >= w)

    if (x == -1 & y == -1):
        x = (W - w) // 2
        y = (H - h) // 2

    if (x, y, w, h) != (0, 0, W, H):
        image = image[y:y + h, x:x + w]
        mask = mask[y:y + h, x:x + w]

    return image, mask


def random_crop_transform(image, mask, w, h, u=0.5):
    """
    :param image: original image
    :param mask: multi_mask
    :param w: width to crop
    :param h: height to crop
    :param u: prob to do crop
    :return:
    """
    x, y = -1, -1
    H, W = image.shape[:2]

    if random.random() < u:
        if H != h:
            y = np.random.choice(H - h)
        else:
            y = 0

        if W != w:
            x = np.random.choice(W - w)
        else:
            x = 0

    return fix_crop_transform(image, mask, x, y, w, h)


def random_horizontal_flip_transform(image, mask, u=0.5):
    if random.random() < u:
        image = cv2.flip(image, 1)  # np.fliplr(img) ##left-right
        mask = cv2.flip(mask, 1)
    return image, mask


def random_vertical_flip_transform(image, mask, u=0.5):
    if random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    return image, mask


def random_rotate90_transform(image, mask, u=0.5):
    if random.random() < u:

        angle = random.randint(1, 3) * 90
        if angle == 90:
            image = image.transpose(1, 0, 2)  # cv2.transpose(img)
            image = cv2.flip(image, 1)
            mask = mask.transpose(1, 0)
            mask = cv2.flip(mask, 1)

        elif angle == 180:
            image = cv2.flip(image, -1)
            mask = cv2.flip(mask, -1)

        elif angle == 270:
            image = image.transpose(1, 0, 2)  # cv2.transpose(img)
            image = cv2.flip(image, 0)
            mask = mask.transpose(1, 0)
            mask = cv2.flip(mask, 0)
    return image, mask


def relabel_multi_mask(multi_mask):
    data = multi_mask
    data = data[:, :, np.newaxis]
    unique_color = set(tuple(v) for m in data for v in m)
    # print(len(unique_color))

    H, W = data.shape[:2]
    multi_mask = np.zeros((H, W), np.int32)
    for color in unique_color:
        # print(color)
        if color == (0,): continue

        mask = (data == color).all(axis=2)
        label = skimage.morphology.label(mask)

        index = [label != 0]
        multi_mask[index] = label[index] + multi_mask.max()

    return multi_mask


def random_shift_scale_rotate_transform(image, mask,
                                        shift_limit=[-0.0625, 0.0625],
                                        scale_limit=[1 / 1.2, 1.2],
                                        rotate_limit=[-15, 15],
                                        borderMode=cv2.BORDER_REFLECT_101,
                                        u=0.5):
    # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT
    if random.random() < u:
        height, width, channel = image.shape

        angle = random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = random.uniform(scale_limit[0], scale_limit[1])
        sx = scale
        sy = scale
        dx = round(random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = math.cos(angle / 180 * math.pi) * (sx)
        ss = math.sin(angle / 180 * math.pi) * (sy)
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode, borderValue=(
            0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

        mask = mask.astype(np.float32)
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST,  # cv2.INTER_LINEAR
                                   borderMode=borderMode, borderValue=(
            0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
        mask = mask.astype(np.int32)
        mask = relabel_multi_mask(mask)

    return image, mask
