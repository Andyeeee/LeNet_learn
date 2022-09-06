import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def main():
    transform = transforms.Compose(  # 预处理方法打包
        [transforms.ToTensor(),  # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
         transforms.Normalize((0.5, 0.5, 0.5), (
         0.5, 0.5, 0.5))])  # ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True,
                                               num_workers=0)  # shuffle=True打乱数据 ，num_workers=0载入数据线程数，windos下只能设置 为0，linuxx下可设置值

    # 10000张验证图片；   train=False测试集
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                             shuffle=False, num_workers=0)  # 每次载入 batch_size=5000张
    val_data_iter = iter(val_loader)  # iter 迭代器
    val_image, val_label = val_data_iter.next()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 查看下数据集，训练用不到
    # def imShow(img):
    #    img = img/2+0.5
    #    npIm = img.numpy()
    #    plt.imshow(np.transpose(npIm,(1,2,0)))  #[channel,height,width] 还原成载入图像的shape[height,width,channel]
    #    plt.show()
    #    #img = torchvision.utils.make_grida(img)
    # imShow( torchvision.utils.make_grid(val_image))

    net = LeNet()  # 实例化模型
    loss_function = nn.CrossEntropyLoss()  # 定义损失函数This criterion combines :class:`~torch.nn.LogSoftmax` and :class:`~torch.nn.NLLLoss` in one single class.
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # net.parameters() 就是LeNet所需要训练的参数

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0  # 累加训练过程的损失
        for step, data in enumerate(train_loader, start=0):  # 循环-->遍历训练集样本
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()  # 将历史损失梯度清零，如果不清除历史梯度，就会对计算历史梯度进行累加（思维转变：相当于我们用了一个很大的batchSize，让效果更好）
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()  # loss反向传播
            optimizer.step()  # step() 实现参数更新

            # print statistics打印过程
            running_loss += loss.item()
            if step % 500 == 499:  # print every 500 mini-batches，每隔500步打印一次
                with torch.no_grad():  # torch.no_grad() 意思：不计算每一个节点的损失梯度，如果不加这个函数，在测试过程中也会计算损失梯度（会消耗算力，资源，会存储每个节点的损失梯度，消耗内存，容易导致程序内存不足而崩溃）
                    outputs = net(val_image)  # [batch, 10]  #正向传播
                    predict_y = torch.max(outputs, dim=1)[
                        1]  # max网络预测类别归属（ dim=1纬度1上寻找最大值，10个节点中寻找最大值），[1] 代表我们只需要index即索引值
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(
                        0)  # .eq(predict_y, val_label).sum()累积预测对的数目是个terson张量； .item()将张量转换为标量; /val_label.size(0)除以样本总量

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))  # running_loss / 500是500步的平均训练误差
                    running_loss = 0.0  # 将累积训练误差清零，再进行下500次的训练，重新累计

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)  # 模型权重参数保存net.state_dict()


if __name__ == '__main__':
    main()

# train_set = torchvision.datasets. 官方在这下面为我们提供了大量数据集，有时间.出来
