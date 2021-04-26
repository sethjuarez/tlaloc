import os
import json
import time
import torch
import logging
import requests
import numpy as np
from pathlib import Path
from data import SeqDataset
from model import EarningsGRUModel
from datetime import datetime, timedelta

meta = None
model: EarningsGRUModel = None
logger = logging.getLogger()

def init():
    global model, meta, logger
    logger.info('Attempt to load model artifacts')
    try:
        from azureml.core.model import Model
        path = Model.get_model_path('earnings-model-1')
        logger.info('Loaded model from AML')
    except:
        path = '../outputs/model'
        logger.info('Using Local')

    logger.info('Resolving artifact paths')
    root_dir = Path(path).resolve()
    trnsf_path = root_dir / 'params.json'
    model_path = root_dir / 'model.pth'
    logger.info(f'metadata path: {trnsf_path}')
    logger.info(f'model path: {model_path}')

    logger.info('loading metadata')
    with open(trnsf_path, 'r') as f:
        meta = json.load(f)
    logger.info(f'metadata load complete: {json.dumps(meta)}')

    logger.info(f'instantiating mode with params: {json.dumps(meta["model"])}')
    model = EarningsGRUModel(**meta['model'])
    logger.info(f'instantiation complete, loading state dictionary')
    model.load_state_dict(torch.load(model_path))
    logger.info(f'model state loaded successfully')
    logger.info(f'init complete!')

def run(raw_data):
    global model, meta, logger
    logger.info('starting inference clock')
    prev_time = time.time()
    
    post = json.loads(raw_data)

    # input data
    logger.info(f'loading post info {json.dumps(raw_data)}')
    sequence = np.array(post['sequence'])
    logger.info('sequence loaded')

     # params
    window = int(meta['data']['window'])
    overlap = int(post['overlap'])
    forward = int(post['forward'])
    logger.info('parameters loaded')
    
    # scale
    cmin = float(meta['data']['min'])
    cmax = float(meta['data']['max'])
    sequence = SeqDataset.scale(sequence, cmin, cmax)
    logger.info(f'scaling with min/max: {cmin}/{cmax}')

    # predict with model
    logger.info('pre-prediction')
    predicted = model.predict(list(sequence), 
                                window, overlap, forward)
    logger.info('prediction complete')
    
    # scale up predicted
    logger.info(f'scaling with min/max: {cmin}/{cmax}')
    predicted = SeqDataset.inverse_scale(torch.FloatTensor(predicted), 
                                            cmin, cmax)

    # calculate time
    logger.info('stopping clock')
    current_time = time.time()
    inference_time = timedelta(seconds=current_time - prev_time)

    logger.info('preparing payload')
    payload = {
        'time': str(inference_time.total_seconds()),
        'prediction': list(predicted.numpy().tolist()),
    }
    logger.info(f'payload: {payload}')
    logger.info('inference complete')
    return payload

if __name__ == '__main__':
    # initialize
    init()

    # input data
    seq = \
      [330429.34, 335124.1 , 337927.4 , 337882.78, 340398.47, 348004.94,
       351887.16, 353909.22, 349526.72, 357322.5 , 360355.3 , 357960.47,
       353047.97, 360072.47, 358046.12, 354156.03, 352541.2 , 355159.06,
       357326.4 , 359204.56, 354851.44, 345909.5 , 344326.1 , 345570.56,
       344388.5 , 346549.12, 344628.03, 345876.88, 342719.56, 338990.  ,
       345393.6 , 346861.25, 350205.9 , 351335.66, 349425.16, 354068.47,
       353115.  , 352033.  , 355203.28, 354767.22, 347558.28, 349151.47,
       348895.66, 345206.62, 342617.56, 344604.34, 349649.1 , 345719.28,
       348129.8 , 352518.56, 352416.53, 354929.84, 349414.72, 348425.  ,
       351704.2 , 351390.56, 344502.47, 336911.22, 333464.03, 330384.1 ,
       329339.9 , 333387.9 , 333351.94, 333792.2 , 333526.6 , 331457.75,
       334915.47, 335476.75, 339905.9 , 336842.47, 335527.6 , 337232.53,
       336993.06, 338816.78, 332234.88, 335720.66, 336146.9 , 341194.94,
       342124.25, 343852.38, 343336.38, 344777.4 , 341068.2 , 344561.62,
       345402.28, 343044.2 , 340084.12, 344043.47, 337525.16, 338148.5 ,
       333078.78, 334860.88, 334035.1 , 334407.75, 330211.56, 329117.44,
       326872.2 , 322658.28, 322081.62, 318807.9 , 317481.3 , 317425.66,
       323189.9 , 322032.94, 319100.  , 311249.  , 305504.3 , 307887.62,
       304874.22, 310079.56, 307963.47, 314782.6 , 318511.66, 314644.7 ,
       314424.28, 314716.22, 316446.66, 317378.34, 316824.  , 317122.38,
       318583.47, 319947.28, 325159.72, 328583.12, 328360.34, 322865.66,
       325752.  , 329994.5 , 333320.44, 339669.94, 341644.03, 339608.72,
       337593.78, 340009.94, 339467.66]

    # run model
    post = { 
        'sequence': seq, 
        'overlap': 90, 
        'forward': 14
    }

    print(json.dumps(post, indent=4))
    o = run(json.dumps(post))
    print(json.dumps(o, indent=4))