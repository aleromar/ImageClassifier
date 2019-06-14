import argparse
import pyutils
import nnutils
import torch

def initTrainParser(parser):
    parser.add_argument(dest='trainDir',action='store', help='Directory where the training data is')
    parser.add_argument('--save_dir', dest='savingDir', action='store', default='./modelSaved', help='Set directory to save checkpoints')
    parser.add_argument('--arch', dest='archmodel', action='store', default = 'vgg16_bn', help='Choose architecture')
    parser.add_argument('--learning_rate', dest='learnRate', action='store', default = 0.001, type=float, help='Learning rate assigned to the optimizer')
    parser.add_argument('-hu', '--hidden_units', nargs='+', type=int, default=[0], help='List containing the number of hidden units for each hidden layer ie: -hu 512 256 128')
    parser.add_argument('--epochs', dest='numEpochs', action='store', default = 1, type=int, help='Number of epochs to use to train the neural network')
    parser.add_argument('--gpu', action="store_true", default=False, help = 'Flag that indicates whether to use gpu for training')
    
#python train.py ./flowers/train --save_dir ./ --arch vgg16_bn --learning_rate 0.0005 -hu 4096 2048 1024 256 --epochs 10 --gpu

if __name__ == "__main__":
    # Process the input parameters into the script
    parser = argparse.ArgumentParser(description='Parsing training data')
    initTrainParser(parser)
    args = parser.parse_args() # archmodel, gpu, hidden_units, learnRate, numEpochs, savingDir, trainDir
    print(args)
    
    # Process the training images so they are ready to be fed into the algorithm
    dataloader, classtoid = pyutils.trainTransforms(args.trainDir)
    print('- Loaded training data from: {}'.format(args.trainDir))
    # Construct the model, composed of a custom classifier and a transfer learning model used as basemodel
    requestedclassifier = nnutils.ClassifierARM(numNodesHidden=args.hidden_units)
    print('- Created a classifier with {} hidden layers with units {}'.format(len(args.hidden_units), args.hidden_units))
    transfermodel = nnutils.transferlearningModel(args.archmodel)
    model, optimizer = pyutils.loadFromScratch(transfermodel,True,requestedclassifier,args.learnRate, classtoid, args.gpu)
    print('- Transfer learning model with custom classifier and Adam optimizer created')
    
    # Run the algorithm with the user specified parameters
    nnutils.runtraining(model,optimizer,args.numEpochs,dataloader,torch.device("cuda:0" if args.gpu else "cpu"))
    
    # Save the model for future usage
    pyutils.savemodel(model,optimizer.state_dict(),args.savingDir+'modelcp2(){}.pth'.format(args.archmodel))
