import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from fastapi import FastAPI, File
import numpy as np
import logging
from predictor import Predictor
from utlis import to_bytes, BytesIO, Image
logging.basicConfig(format='[%(levelname).1s %(asctime)s %(module)s:%(lineno)d@%(funcName)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
predictor = Predictor()
app = FastAPI()


@app.post("/stylize/")
def stylize(content_image: bytes = File(...), style_image: bytes = File(...)):
    """

    :param content_image: image which content should be used in generated one
    :param style_image: image which style should be used in generated one
    :return: resulting image in bytes array with base64 encoding
    """
    result = None
    try:
        content_image = np.array(Image.open(BytesIO(content_image)))
        style_image = np.array(Image.open(BytesIO(style_image)))
        result = predictor.predict(content_image, style_image)[0]
        result = to_bytes(result)
        logging.info('Made stylization.')
    except Exception as e:
        logging.error('Error : ' + str(e))
    return result
