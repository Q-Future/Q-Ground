import numpy as np
import os
from PIL import Image
from PIL import ImageColor


# [120, 120, 120], 
PALETTE = [[180, 120, 120], [6, 230, 230], [80, 50, 50],
           [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
           [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
           [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
           [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
           [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
           [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
           [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
           [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
           [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
           [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
           [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
           [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
           [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
           [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
           [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
           [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
           [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
           [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
           [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
           [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
           [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
           [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
           [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
           [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
           [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
           [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
           [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
           [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
           [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
           [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
           [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
           [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
           [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
           [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
           [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
           [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
           [102, 255, 0], [92, 0, 255]]


def is_image_file(filename):
    return any(
        filename.lower().endswith(extension)
        for extension in Image.registered_extensions()
    )


def scandir_images(dir, max_dataset_size=float('inf'), followlinks=True):
    """Get all image files from a directory and return a sorted list of fullpath."""
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=followlinks)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return sorted(images[: min(max_dataset_size, len(images))])


def masks2rgb(masks):
    mask_rgb = np.zeros(masks[0].shape + (3,), dtype=np.uint8)
    for i, mask in enumerate(masks):
        mask_rgb[mask == 1] = PALETTE[i]
    return mask_rgb


DISTORTION_TYPES = {
    'jitter': np.array(ImageColor.getrgb('#4b54e1')),
    'noise': np.array(ImageColor.getrgb('#93fff0')),
    'overexposure': np.array(ImageColor.getrgb('#cde55d')),
    'blur': np.array(ImageColor.getrgb('#e45c5c')),
    'low light': np.array(ImageColor.getrgb('#35e344')), 
}

DISTORTION_TYPES_EXT = {
    'jitter': np.array(ImageColor.getrgb('#4b54e1')),
    'noise': np.array(ImageColor.getrgb('#93fff0')),
    'overexposure': np.array(ImageColor.getrgb('#cde55d')),
    'blur': np.array(ImageColor.getrgb('#e45c5c')),
    'low light': np.array(ImageColor.getrgb('#35e344')), 
    'no distortion': np.array([0, 0, 0])
}


def rgb2label(mask_rgb):
    mask_label = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    for i, (k, v) in enumerate(DISTORTION_TYPES.items()):
        mask_label[np.isclose(mask_rgb, v, rtol=0.2, atol=20).all(axis=-1)] = i + 1
    return mask_label


def label2rgb(label):
    mask_rgb = np.zeros(label.shape + (3,), dtype=np.uint8)
    for i, (k, v) in enumerate(DISTORTION_TYPES.items()):
        mask_rgb[label == i + 1] = v
    return mask_rgb


def get_distortion_mask(seg_map, category_name):
    mask = np.zeros_like(seg_map)
    mask[np.isclose(seg_map, DISTORTION_TYPES[category_name], rtol=0.2, atol=20)] = 1 
    mask = np.prod(mask, axis=-1)
    if mask.sum() > 100:    # mask region > 100 pixels
        return mask.astype(np.uint8)
    else:
        return None 


def get_all_distortion_masks(seg_map):
    masks = {}
    for category_name in DISTORTION_TYPES.keys():
        mask = get_distortion_mask(seg_map, category_name)
        if mask is not None:
            masks[category_name] = mask
    return masks


if __name__ == '__main__':
    tmp_org_mask_save_dir = './tmp_test/org_masks/'
    tmp_rec_mask_save_dir = './tmp_test/rec_masks/'

    diff = 0
    for idx, img_path in enumerate(scandir_images('./qground_dataset/masks/spaq_koniq/')):
        org_mask = Image.open(img_path)
        org_mask.save(os.path.join(tmp_org_mask_save_dir, os.path.basename(img_path)))
        org_mask_label = rgb2label(np.array(org_mask))
        rec_mask = label2rgb(org_mask_label)
        Image.fromarray(rec_mask).save(os.path.join(tmp_rec_mask_save_dir, os.path.basename(img_path)))

        diff = np.mean((np.array(org_mask) - rec_mask) ** 2)
        print(f"idx: {idx}, rec PSNR: {diff}")

        if idx > 50:
            break
