import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from fastapi import FastAPI
import numpy as np
import logging
from app.predictor import Predictor
from app.utlis import to_bytes, BytesIO, Image
from pydantic import BaseModel
import base64

logging.basicConfig(format='[%(levelname).1s %(asctime)s %(module)s:%(lineno)d@%(funcName)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
predictor = Predictor()
app = FastAPI()


class Body(BaseModel):
    content_image: str
    style_image: str


@app.post("/stylize/")
def stylize(body: Body):
    """

    - :param content_image: image which content should be used in generated one
    - :param style_image: image which style should be used in generated one


    :return: resulting image in bytes array with base64 encoding
    """
    result = None
    content_image, style_image = body.content_image, body.style_image
    try:
        content_image = np.array(Image.open(BytesIO(base64.b64decode(content_image))))
        logging.info('Shape of content image : {}'.format(content_image.shape))
        style_image = np.array(Image.open(BytesIO(base64.b64decode(style_image))))
        logging.info('Shape of style image : {}'.format(content_image.shape))
        logging.info('Got content and style images!')
        result = predictor.predict(content_image, style_image)[0]
        result = to_bytes(result)
        logging.info('Made stylization.')
    except Exception as e:
        logging.error('Error : ' + str(e))
    logging.info('Number of bytes to return : {}'.format(len(result)))
    return result
