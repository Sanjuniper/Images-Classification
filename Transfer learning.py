
from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns
# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import pandas as pd
import os

# Image manipulations
from PIL import Image
# Useful for examining network
from torchsummary import summary
# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt
import VGG
plt.rcParams['font.size'] = 14

# Printing out all outputs
InteractiveShell.ast_node_interactivity = 'all'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

datadir = 'E:/class_sortrubbish/'
traindir = datadir + 'class-train-data/'
validdir = datadir + 'class-valid-data/'
testdir = datadir + 'class-test-data/'

save_file_name = 'vgg16-transfer-4.pt'
checkpoint_path = 'vgg16-transfer-4.pth'
save_resfile_name = 'res50-transfer-4.pt'
check_res_point_path = 'res50-transfer-4.pth'
# Change to fit hardware
batch_size = 1

# Whether to train on a gpu
train_on_gpu = cuda.is_available()
print('Train on gpu: {}'.format(train_on_gpu))

categories = []
img_categories = []
n_train = []
n_valid = []
n_test = []
hs = []
ws = []

# Iterate through each category    os.listdir('dirname')：列出指定目录下的所有文件和子目录，包括隐藏文件，并以列表方式打印
for d in os.listdir(traindir):
    categories.append(d)

    # Number of each image
    train_imgs = os.listdir(traindir + d)
    valid_imgs = os.listdir(validdir + d)
    test_imgs = os.listdir(testdir + d)
    n_train.append(len(train_imgs))
    n_valid.append(len(valid_imgs))
    n_test.append(len(test_imgs))

    # Find stats for train images
    for i in train_imgs:
        img_categories.append(d)
        img = Image.open(traindir + d + '/' + i)
        img_array = np.array(img)
        # Shape
        hs.append(img_array.shape[0])
        ws.append(img_array.shape[1])

# Dataframe of categories  #使用字典创建数据结构
cat_df = pd.DataFrame({'category': categories,
                           'n_train': n_train,
                           'n_valid': n_valid, 'n_test': n_test}). \
        sort_values('category')

# Dataframe of training images
image_df = pd.DataFrame({
        'category': img_categories,
        'height': hs,
        'width': ws
    })

# Image transformations
image_transforms = {
    # Train uses data augmentation:数据增强（torchvision提供），可改进模型的泛化能力。
    'train':
    transforms.Compose([       #Transfoms 是很常用的图片变换方式，可以通过compose将各个变换串联起来
        transforms.Resize((224,224)),  #将输入的PIL图片转换成给定的尺寸的大小
        transforms.RandomRotation(degrees=15),   #按角度旋转图像。
        transforms.ColorJitter(),     #随机改变图片的亮度、对比度和饱和度。Grayscale(num_output_channels=1)将图片转成灰度图。参数：　num_output_channels(int) ——　(1或者3)，输出图片的通道数量；返回：　输入图片的灰度图，如果num_output_channels=1, 返回的图片为单通道. 如果 num_output_channels=3, 返回的图片为3通道图片，且r=g=b；返回类型：PIL图片类型
        transforms.RandomHorizontalFlip(),   #以给定的概率随机水平翻转给定的PIL图像
          # Image net standards
        transforms.ToTensor(),  #将PIL图片或者numpy.ndarray转成Tensor类型的
        transforms.Normalize([0.485, 0.456, 0.406],   #用均值和标准差对张量图像进行标准化处理
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Datasets from each folder  一个通用的数据加载器，数据集中的数据以以下方式组织：root/dog/xxx.png
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'val':
    datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
}
# 如何获得 dataloader，答案是将自定义的 dataset 传入torch.utils.data.DataLoader中。
# dataloader 是 python 中的可迭代对象，我们可以通过 for 循环将数据一一取出。
# Dataloader iterators
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
}

trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)
#142


n_classes = len(cat_df)
print('There are {} different classes.'.format(n_classes))

len(data['train'].classes)

#pre-trained model
def get_pretrained_model(model_name):


    if model_name == 'vgg16':
        model = models.vgg16()
        model.load_state_dict(torch.load('E:/class_sortrubbish/vgg16-397923af.pth'))  #model.load_state_dict(torch.load(PATH))恢复模型中的参数：使用这种方法，我们需要自己导入模型的结构信息。

        # Freeze early layers
        for param in model.parameters():  #对原模型中的参数进行遍历操作
            param.requires_grad = False   #将参数中的param.requires_grad全部设置为False，这样对应的参数将不计算梯度，当然也不会进行梯度更新了，这就是之前说到的冻结操作；
        n_inputs = model.classifier[6].in_features   #输入个数根据分类文件夹个数而定。

        # Add on classifier
        # 然后，定义新的全连接层结构并重新赋值给model.classifier。（考虑到输出结果个数，需修改全连接层）
        #在完成了新的全连接层定义后，全连接层中的parma.requires_grad参数会被默认重置为True，所以不需要再次遍历参数来进行解冻操作。
        #pytorch使用torch.nn.Sequential快速搭建神经网络：torch.nn.Sequential是一个顺序容器，模块将按照构造函数中传递的顺序添加到模块中。另外，也可以传入一个有序模块。
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
            nn.LogSoftmax(dim=1))

    elif model_name == 'resnet50':
        #model = models.resnet50(pretrained=True)
        model = models.resnet50()
        model.load_state_dict(torch.load('E:/class_sortrubbish/resnet50-19c8e357.pth'))

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
    # elif model_name == 'vggF1':
    #
    #     model = VGG.vggF1()
    #     model11 = models.vgg11()
    #     model11.load_state_dict(torch.load('E:/fzz/myclass/vgg11-bbd30ac9.pth'))
    #     print('------model11------------')
    #     print(model11)
    #     print('-------model-------------')
    #     print(model)
    #     print('-------model11.state_dict().items()---')
    #     print(model11.state_dict().items())
    #     print('-------model.state_dict().items()---')
    #     print(model.state_dict().items())
    #     print('-------model11.state_dict()---')
    #     print(model11.state_dict())
    #     print('-------model.state_dict()---')
    #     print(model.state_dict())
    #     pretrained_dict = model11.state_dict()   #返回内置预训练model11的字典
    #
    #     model_dict = model.state_dict()  #返回我们自己model的字典
    #
    #     # ------------------------最关键的三步------------------------------------------
    #     # 1. filter out unnecessary keys，也就是说从内置模块中删除掉我们不需要的字典
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     # 2. overwrite entries in the existing state dict，利用pretrained_dict更新现有的model_dict
    #     model_dict.update(pretrained_dict)
    #     # 3. load the new state dict，更新模型，加载我们真正需要的state_dict
    #     model.load_state_dict(model_dict)
    #     # Freeze early layers
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     n_inputs = model.classifier[6].in_features

        # Add on classifier
        '''model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
        '''
        # model.classifier[6] = nn.Sequential(
        #     nn.Linear(n_inputs, 2048), nn.ReLU(), nn.Dropout(0.4),
        #     #nn.AdaptiveAvgPool2d((n_inputs, 2048)), nn.ReLU(), nn.Dropout(0.4),
        #     #nn.AdaptiveAvgPool1d(256),nn.ReLU(), nn.Dropout(0.4),
        #     nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.4),
        #     #nn.AdaptiveAvgPool1d(n_classes), nn.LogSoftmax(dim=1))
        #     nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
    # Move to gpu and parallelize
    if train_on_gpu:
        model = model.to('cuda')

    return model
#
model = get_pretrained_model('vgg16')
#  print params
#print(summary(model, input_size=(3, 128, 128), batch_size=batch_size, device='cuda'))
print(model)


#Mapping of Classes to Indexes
model.class_to_idx = data['train'].class_to_idx
model.idx_to_class = {
    idx: class_
    for class_, idx in model.class_to_idx.items()
}
cls = []
cls = list(model.idx_to_class.items())[:10]
#print(cls)


# loss and optimizer 定义好模型的损失函数和对参数进行优化的优化函数
criterion = nn.NLLLoss()   #Negative Log Liklihood(NLL) Loss:负对数似然损失函数。CrossEntropyLoss()=log_softmax() + NLLLoss()，CrossEntropyLoss() 与 NLLLoss()类似, 唯一的不同是它为我们去做 softmax并取对数.
optimizer = optim.Adam(model.parameters(), lr=0.0001,)

#train 在代码中优化函数使用的是Adam，损失函数使用的是负对数似然，训练次数总共是20次
def train(model,    #模型
          criterion,  #损失函数
          optimizer,  #优化函数
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,   #训练次数总共是20次
          print_every=2):

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
                #print("data:",data)
                #print("target:", target.data)
            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities

            output = model(data)



            # Loss and backpropagation of gradients
            print('target',target)
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history




model, history = train(
    model,
    criterion,
    optimizer,
    dataloaders['train'],
    dataloaders['val'],
    save_file_name=save_file_name,
    max_epochs_stop=5,
    n_epochs=30,
    print_every = 2

       )
'''
def save_checkpoint(model, path):

    print('model ready to save')
    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Path must have the correct model name"

    # Basic details
    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'epochs': model.epochs,
    }

    # Extract the final classifier and the state dictionary
    if model_name == 'vgg16':
        # Check to see if model was parallelized
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'resnet50':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)
    print('model already saved')
print('=======================')

plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(
        history[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Negative Log Likelihood')
plt.title('Training and Validation Losses')

plt.figure(figsize=(8, 6))
for c in ['train_acc', 'valid_acc']:
    plt.plot(
        100 * history[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Accuracy')
plt.title('Training and Validation Accuracy')
plt.show()
plt.savefig('acc.png')
print('ready to save')
save_checkpoint(model, path=checkpoint_path)
print('already saved')
'''




