from utlis import *


class Predictor:
    def __init__(self, link_model='https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2',
                 style_image_size=(256, 256),
                 content_image_size=(384,384)):
        self.link_model = link_model
        self.model = hub.load(self.link_model)
        self.style_image_size = style_image_size
        self.content_image_size = content_image_size

    def predict(self, content_image, style_image):
        content_processed = process(content_image, self.content_image_size)
        style_processed = process(style_image,  self.style_image_size)
        result = self.model(tf.constant(content_processed), tf.constant(style_processed))[0].numpy()
        result*=255
        result = result.astype('uint8')
        return result
