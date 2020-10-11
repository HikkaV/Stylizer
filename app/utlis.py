import tensorflow as tf
import tensorflow_hub as hub
from io import BytesIO
from PIL import Image
import numpy as np
import base64


def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image


def to_bytes(image):
    with BytesIO() as output:
        with Image.fromarray(image) as img:
            img.convert('RGB').save(output, 'BMP')
        data = output.getvalue()
    return base64.b64encode(data)


def process(image, image_size=(256, 256)):
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    image = image.astype(np.float32)[np.newaxis, ...]
    if image.max() > 1.0:
        image = image / 255.
    if len(image.shape) == 3:
        image = tf.stack([image, image, image], axis=-1)
    image = crop_center(image)
    image = tf.image.resize(image, image_size, preserve_aspect_ratio=True)
    return image
