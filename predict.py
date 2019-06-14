import argparse
import nnutils
import pyutils
import pandas as pd
import torch
from PIL import Image
import numpy as np

def initPredictParser(parser):
    parser.add_argument(dest='imageDir',action='store', help='Path to image')
    parser.add_argument(dest='checkpoint',action='store', help='Model checkpoint to load')
    parser.add_argument('--top_k', dest='topK', action='store', default=3, type=int, help='returns K most probable classes')
    parser.add_argument('--category_names', dest='catnames', action='store', default = '', help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action="store_true", default=False, help = 'Flag that indicates whether to use gpu for training')

# python predict.py ./flowers/test/15/image_06351.jpg 'modelcp()vgg16_bn.pth' --top_k 3 --category_names cat_to_name.json --gpu
if __name__ == "__main__":
    # Process the input parameters into the script
    parser = argparse.ArgumentParser(description='Parsing predict args')
    initPredictParser(parser)
    args = parser.parse_args() # imageDir, checkpoint, topK, catnames, gpu
    print(args)
    
    # Build a model upon which we'll load the pre-saved checkpoint
    transfermodel = nnutils.transferlearningModel( args.checkpoint.split('()')[1].split('.')[0] )
    model, _ = pyutils.loadModelFromFile(transfermodel, args.checkpoint, args.gpu)
    
    # Process the image the user wants to classify, and make a prediction
    im = Image.open(args.imageDir)
    probs,classes = nnutils.predict(im,model,args.topK, args.gpu)
    
    # Process prediction results so we output the information that is expected
    df = pd.DataFrame(columns=['probs'],index=classes,data=probs)
    mapcattoname = pyutils.readCatToName(args.catnames)
    df.index = df.index.map(mapcattoname)
    print(df.head())