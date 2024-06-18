import torch
from model_ssconv import resnet50
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import datetime
from tqdm import tqdm
import argparse
from torch import nn
import thop

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('cuda is available : ', torch.cuda.is_available())

parser = argparse.ArgumentParser(description='Train CIFAR10 with PyTorch')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

##########################
#          Data          #soos
##########################

print('------ Preparing data ------')

batch_size = 16

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR10(
    root='./CIFAR10', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.CIFAR10(
    root='./CIFAR10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=0)




best_acc = 0
start_epoch = 1
end_epoch = start_epoch + 51

net = resnet50()
model_weight_path = "./resnet50-pre.pth"
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
pre_weights = torch.load(model_weight_path, map_location='cpu')
del_key = []
for key, _ in pre_weights.items():
    if "conv2" in key:
        del_key.append(key)

for key in del_key:
    del pre_weights[key]
net.load_state_dict(pre_weights, strict=False)
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 10)
net.to(device)

if args.resume:
    print('------ Loading checkpoint ------')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    end_epoch += start_epoch

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)


##########################
#     Model Summary      #
##########################


def count_parameters(model):
    # 打印参数数量
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    input_size = (1, 3, 32, 32)  # 假设的输入大小
    dummy_input = torch.randn(*input_size).to(device)

    # 计算FLOPs和参数数量
    flops, params = thop.profile(model, inputs=(dummy_input,))

    # 打印FLOPs和参数数量
    print(f'FLOPs: {flops / 1e9:.2f} G')  # 将FLOPs转换为GigaFLOPs（GFLOPs）
    print(f'Number of parameters: {params:,}')



##########################
#        Training        #
##########################

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs).to(device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


        if batch_idx % 50 == 0:
            print('\tLoss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    train_loss = train_loss / len(train_loader)
    train_acc = 100. * correct / total

    print('\n', time.asctime(time.localtime(time.time())))
    print(' Epoch: %d | Train_loss: %.3f | Train_acc: %.3f%% \n' % (epoch, train_loss, train_acc))

    return train_loss, train_acc


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs).to(device)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                print('\tLoss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total

        print('\n', time.asctime(time.localtime(time.time())))
        print(' Epoch: %d | Test_loss: %.3f | Test_acc: %.3f%% \n' % (epoch, test_loss, test_acc))

    if test_acc > best_acc:
        print('------ Saving model------')
        state = {
            'net': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/model_%d_%.3f.pth' % (epoch, best_acc))
        best_acc = test_acc

    return test_loss, test_acc



def save_csv(epoch, save_train_loss, save_train_acc, save_test_loss, save_test_acc):
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    step = 'Step[%d]' % epoch
    train_loss = '%f' % save_train_loss
    train_acc = '%g' % save_train_acc
    test_loss = '%f' % save_test_loss
    test_acc = '%g' % save_test_acc

    print('------ Saving csv ------')
    data_row = [time, step, train_loss, train_acc, test_loss, test_acc]

    # 创建DataFrame来保存当前行的数据
    data = pd.DataFrame([data_row], columns=['time', 'step', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    # 检查文件是否存在
    if not os.path.exists('./train_acc.csv') or epoch == 1:  # 如果是第一次保存
        mode = 'w'  # 使用写入模式
    else:
        mode = 'a'  # 使用追加模式

    # 保存DataFrame到CSV文件
    with open('./train_acc.csv', mode) as f:
        data.to_csv(f, header=False if epoch != 1 else True, index=False)  # 如果不是第一次保存，则不包含列名



def draw_acc():
    filename = './train_acc.csv'

    train_data = pd.read_csv(filename)
    print(train_data.head())

    Epoch = list(range(1, len(train_data) + 1))

    train_loss = train_data['train_loss'].astype(float)
    train_accuracy = train_data['train_acc'].astype(float)
    test_loss = train_data['test_loss'].astype(float)
    test_accuracy = train_data['test_acc'].astype(float)

    plt.plot(Epoch, train_loss, 'g-.', label='Train Loss')
    plt.plot(Epoch, train_accuracy, 'r-', label='Train Accuracy')
    plt.plot(Epoch, test_loss, 'b-.', label='Test Loss')
    plt.plot(Epoch, test_accuracy, 'm-', label='Test Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Loss & Accuracy')
    plt.yticks([j for j in range(0, 101, 10)])  # 注意这里的范围可能需要根据实际情况调整
    plt.title('Epoch -- Loss & Accuracy')

    plt.legend(loc='upper right', fontsize=8, frameon=False)  # 稍微调整了legend的位置
    plt.show()




if __name__ == '__main__':

    count_parameters(net)
    
    for epoch in range(start_epoch, end_epoch):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        scheduler.step()
    
        save_csv(epoch, train_loss, train_acc, test_loss, test_acc)

    draw_acc()
