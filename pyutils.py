from torchvision import datasets, transforms, models
import torch
import os,copy
from PIL import Image
import numpy as np
import nnutils
import json
from torch.utils.data.sampler import SubsetRandomSampler

def readCatToName(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def trainTransforms(data_dir):
    data_transforms = {'train':transforms.Compose([transforms.RandomRotation(30),
                               transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    dataloading = {'train': datasets.ImageFolder(data_dir, transform=data_transforms['train'])}
    
    trainingindexes = list(np.random.choice(np.arange(0,len(dataloading['train'])), int(len(dataloading['train'])*0.8),replace=False))
    testindexes = [i for i in range(len(dataloading['train'])) if i not in trainingindexes ]
    
    train_sampler = SubsetRandomSampler(trainingindexes)
    test_sampler = SubsetRandomSampler(testindexes)
    
    dataloaders = {'train': torch.utils.data.DataLoader(dataloading['train'], batch_size=64, sampler=train_sampler),
                  'test': torch.utils.data.DataLoader(dataloading['train'], batch_size=64, sampler=test_sampler)}

    return dataloaders, dataloading['train'].class_to_idx

def loadBaseModel(model,trained):
    print('Fetching transfer learning model with option "pretrained"= {}'.format(trained))
    return model(pretrained=trained)

def freezeModelState(model):
    print('Freezing base model params. Including the classifier')
    for param in model.parameters():
        param.requires_grad = False
        
def loadFromScratch(model,trained,classifier,learnr, classtoidx, gpuflag=False):
    device = torch.device("cuda:0" if gpuflag else "cpu")
    print('Loading model from scratch')
    basemodel = loadBaseModel(model,trained)
    # Freezing the base model before loading the classifier
    freezeModelState(basemodel)
    # Load the classifier now to avoid doing a grad required false 
    basemodel.classifier = classifier
    basemodel.classifier.class_to_idx = classtoidx
    basemodel.classifier.num_epochs = 0
    # The model should be moved to the device before the optimizer is created
    basemodel.to(device)
    optimizer = nnutils.findOptimizer(basemodel,learnr) 
    return basemodel,optimizer

def savemodel(model,optstatedict,filename):
    print('Saving model to file {}'.format(filename))
    checkpoint = {'input_size': model.classifier.hidden_layers[0].in_features,
                  'output_size': model.classifier.output.out_features,
                  'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
                  'dropoutp': model.classifier.dropout.p,
                  'state_dict': model.state_dict(),
                  'opt_state_dict':optstatedict,
                  'num_epochs': model.classifier.num_epochs,
                  'classToIdx':model.classifier.class_to_idx}
    
    torch.save(checkpoint, filename)
    
def loadModelFromFile(basemodel, filename, gpuflag):
    print('Loading model from file: {}'.format(filename))
    device = torch.device("cuda:0" if gpuflag else "cpu")
    # This map_location option is needed when saving models in one device and loading them
    # in another. For example training on cuda and loading on cpu
    checkpoint = torch.load(filename,map_location=lambda storage, loc: storage)
    # Fetch checkpoint info
    nh = checkpoint['hidden_layers']
    insize = checkpoint['input_size']
    outsize = checkpoint['output_size']
    dropout = checkpoint['dropoutp']
    # Freezing the base model before loading the classifier
    basemodel = loadBaseModel(basemodel,False)
    freezeModelState(basemodel)
    # Load the classifier now to avoid doing a grad required false
    basemodel.classifier = nnutils.ClassifierARM(numInput = insize, numOutput= outsize, dropoutp=dropout,numNodesHidden=nh)
    # Now the model structure should be the same as it was when saved. Then load the state dict to the current model
    basemodel.load_state_dict(checkpoint['state_dict']) 
    basemodel.classifier.num_epochs = checkpoint['num_epochs']
    basemodel.classifier.class_to_idx = checkpoint['classToIdx']
    # The model should be moved to the device before the optimizer is created
    # Otherwise when feeding parameters() to the optimizer, it will think they are
    # under the cpu. Then moving the model to cuda won't inform the optmizer about this change
    # https://pytorch.org/docs/master/optim.html and https://github.com/pytorch/pytorch/issues/7321
    print('Moving model to Device {}'.format(device))
    basemodel.to(device)
    optimizer = nnutils.findOptimizer(basemodel,0.0005)
    # Same as with model, load optimizer's state dict to reload saved state
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    return basemodel, optimizer    

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # Resize 256
    dim1,dim2 = image.size
    factor = 256/dim2 if dim1 > dim2 else 256/dim1
    resizeSize = int(dim1*factor),int(dim2*factor)
    image = image.resize(resizeSize)
    # Crop
    dim1Cen, dim2Cen = int(image.size[0]/2), int(image.size[1]/2)
    image = image.crop((dim1Cen-112,dim2Cen-112,dim1Cen+112,dim2Cen+112))
    # Normalize
    np_image = np.array(image)/255
    for i,(mu,stddev) in enumerate(zip([0.485,0.456,0.406],[0.229,0.224,0.225])):
        np_image[:,:,i] = (np_image[:,:,i] - mu)/stddev
    # Order
    return np_image.transpose((2,0,1))
    