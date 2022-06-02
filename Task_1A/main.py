
# Homecoming (eYRC-2018): Task 1A
# Build a Fully Connected 2-Layer Neural Network to Classify Digits

# NOTE: You can only use Tensor API of PyTorch

from nnet import model
import torch
import matplotlib.pyplot as plt

# TODO: import torch and torchvision libraries
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# We will use torchvision's transforms and datasets

transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('mnist/',train=True,transform=transformation, download=True)
# TODO: Defining torchvision transforms for preprocessing
# TODO: Using torchvision datasets to load MNIST
# TODO: Use torch.utils.data.DataLoader to create loaders for train and test
# NOTE: Use training batch size = 4 in train data loader.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,  batch_size=4, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transforms.ToTensor(),download = True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=4, shuffle=False)                 


# NOTE: Don't change these settings
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# NOTE: Don't change these settings
# Layer size
N_in = 28 * 28 # Input size
N_h1 = 256 # Hidden Layer 1 size
N_h2 = 256 # Hidden Layer 2 size
N_out = 10 # Output size
# Learning rate
lr = 0.15


# init model
net = model.FullyConnected(N_in, N_h1, N_h2, N_out, device=device)
N_epoch = 10 # Or keep it as is
for i in range(0, N_epoch):
    cnt = 0
    total=0
    correct=0
    x=0
    creloss=0
    cost=[]
    index=[]
    for i,(images, labels) in enumerate(train_loader):
        
        images = images.view(images.shape[0],-1)

        creloss,accuracy,outputs=net.train( images, labels , lr, debug= False )
        cost.append(creloss)
        index.append(i)
        score,idx=outputs.max(-1)
        total += labels.size(0)
        correct += (idx == labels).sum().item()
        creloss+=creloss
        x+=1
    print('Test Accuracy : {} %'.format(100 * correct / total))
    print('creloss : {} '.format(creloss / x))
    plt.plot(index,cost)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("N0. of iteration")
    plt.ylabel("loss")
    plt.show()


        
        
    count=0
    total=0
    correct=0
    x=0
    creloss=0
    cost=[]
    index=[]
    for i,(images, labels) in enumerate(train_loader):

        images = images.view(images.shape[0],-1)
        target=labels
        creloss,accuracy,outputs=net.eval( images, target,debug=False)
        cost.append(creloss)
        index.append(i)
        score,idx=outputs.max(-1)

        correct += (idx == labels).sum().item()
        total += labels.size(0)
        creloss+=creloss
        x+=1
    print('Test Accuracy of eval: {} %'.format(100 * correct / total))
    print('creloss of the eval: {} '.format(creloss / x))
    plt.plot(index,cost)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("N0. of iteration")
    plt.ylabel("loss")
    plt.show()
    #print("creloss of eval",creloss,"accuracy of eval",(idx/labels)*100)


cnt=0
total=0
for i,(images, labels) in enumerate(test_loader):
    
    #data= Variable(images,volatile=True)
    #data = data.view(-1, 28 * 28)
    classes=[0,1,2,3,4,5,6,7,8,9]
    images = images.view(images.shape[0],-1)
    score,idx=net.predict(images)
    tmp = []
    for i in labels:
        tmp.append(i.item())
    cntr = 0
    for i in idx:
        print("actual: ",classes[i.item()],"\t Predicted: ",classes[tmp[cntr]])
        if i.item() == tmp[cntr]:
            cnt+=1
        cntr+=1
    total+=4
print('Test Accuracy of test: {} %'.format(100 * cnt/ total))




    



# TODO: Training and Validation Loop
# >>> for n epochs
## >>> for all mini batches
### >>> net.train(...)
## at the end of each training epoch
## >>> net.eval(...)

# TODO: End of Training
# make predictions on randomly selected test examples
# >>> net.predict(...)
