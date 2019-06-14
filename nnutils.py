import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
import time
import pyutils

class ClassifierARM(nn.Module):
    def __init__(self,numInput = 25088, numOutput = 102,dropoutp=0.2,numNodesHidden=[4096,1024]):
        super().__init__()
        numNodes = [numInput] + numNodesHidden + [numOutput]
        # This needs to be added to module list so it appears when printing the model
        self.hidden_layers = nn.ModuleList()
        # Create the hidden layers based on input params
        for layerIn,layerOut in zip(numNodes[:-2],numNodes[1:-1]):# 1 to start from 1 plus 1 for the output layer
            self.hidden_layers.append(nn.Linear(layerIn,layerOut))
        self.output = nn.Linear(numNodes[-2],numNodes[-1])
        self.dropout = nn.Dropout(p=dropoutp)
        # These next to properties are needed when saving the model, so default initialize here
        self.num_epochs = 0
        self.class_to_idx = None
        
    def forward(self,x):
        # Reshape the incoming data to the input layer's dimensions
        x = x.view(x.shape[0],-1)
        # Forward propagation using relu as activation functions
        for fc in self.hidden_layers:
            x = F.relu(fc(x))
            x = self.dropout(x)
        # Finally, apply log softmax so probabilities are just the exponential of the output
        x = F.log_softmax(self.output(x),dim=1)
        return x

def transferlearningModel(sName):
    modelDict = {'vgg11':models.vgg11,
                'vgg11_bn':models.vgg11_bn,
                'vgg13':models.vgg13,
                'vgg13_bn':models.vgg13_bn,
                'vgg16':models.vgg16,
                'vgg16_bn':models.vgg16_bn,
                'vgg19':models.vgg19,
                'vgg19_bn':models.vgg19_bn,
                'resnet18':models.resnet18,
                'resnet34':models.resnet34,
                'resnet50':models.resnet50,
                'resnet101':models.resnet101,
                'resnet152':models.resnet152,
                'densenet121':models.densenet121,
                'densenet169':models.densenet169,
                'densenet161':models.densenet161,
                'densenet201':models.densenet201}
    return modelDict[sName]

def findOptimizer(basemodel,learnr):    
    return optim.Adam(basemodel.classifier.parameters(),lr=learnr)

def nnaccuracy(output,labels):
    probs = torch.exp(output)
    top_p, top_class = probs.topk(1, dim=1)
    hits = top_class.squeeze() == labels.squeeze()
    return torch.mean(hits.type(torch.FloatTensor))

def runNn(optionstring,dataloaders,model,criterion,optimizer,device):
    # This function runs the neural network for training and testing purposes
    # When training it does the forward and backward pass
    # When testing it only executes the forward pass
    accuracy,accumulated_error,retacc,reterr = (0 for _ in range(4))
    if optionstring == 'train':
        for index,(images,labels) in enumerate(dataloaders[optionstring]):
            model.train()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            accuracy += nnaccuracy(output,labels)
            accumulated_error += loss.item()
    elif optionstring in ['test']:
        with torch.no_grad():
            model.eval()
            for index,(images,labels) in enumerate(dataloaders[optionstring]):
                images, labels = images.to(device), labels.to(device)
                output = model.forward(images)
                loss = criterion(output,labels)
                accumulated_error += loss.item()
                accuracy += nnaccuracy(output,labels)
    return accuracy/len(dataloaders[optionstring]),accumulated_error/len(dataloaders[optionstring])

def runtraining(model,optimizer,num_epochs,dataloaders,device):
    model.train()
    epoch_time,train_time,validate_time,train_error,valid_error,train_acc,valid_acc = ([] for _ in range(7))
    criterion = nn.NLLLoss()
    for epoch in range(num_epochs):
        # Train the network and profile timing
        st_epoch = time.time()
        acc,err = runNn('train',dataloaders,model,criterion,optimizer,device)
        # Save important info to relevant lists
        train_time.append(time.time()-st_epoch),train_acc.append(acc),train_error.append(err)
        st_valid = time.time()
        acc_valid,err_valid = runNn('test',dataloaders,model,criterion,optimizer,device)
        validate_time.append(time.time()-st_valid),valid_error.append(err_valid),valid_acc.append(acc_valid),epoch_time.append(time.time()-st_epoch)
                        
        # Output some metrics
        stringToOutput = "Epoch #{} Train Error: {}, Train Accuracy: {}".format(epoch,train_error[-1],train_acc[-1])
        print(stringToOutput)
        stringToOutput = "Epoch #{} Test Error: {}, Test Accuracy: {}".format(epoch,valid_error[-1],valid_acc[-1])
        print(stringToOutput)
        model.classifier.num_epochs += 1

def predict(image_path, model, topk, gpuflag):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda:0" if gpuflag else "cpu")
    im = pyutils.process_image(image_path)
    im = torch.from_numpy(im).float().unsqueeze_(0) # unsqueeze to simulate a batch
    with torch.no_grad():
        model.eval()
        im = im.to(device)
        output = model.forward(im)
        probs = torch.exp(output)
        p,idxes = probs.topk(topk, dim=1)
        p,idxes = p.squeeze().tolist(),idxes.squeeze().tolist()
        idx_to_class = dict([[v,k] for k,v in model.classifier.class_to_idx.items()])
        classes = [idx_to_class[i] for i in idxes]
        return p,classes        