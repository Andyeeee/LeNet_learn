import torch
import torchvision.transforms as transforms
from PIL import Image  #这个模块载入图片

from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), #缩放
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))

    im = Image.open('dog.jpg')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # 在转换为 Python Tensor 的通道排序:[N, C, H, W]; dim=0 在最前面增加维度

    with torch.no_grad(): #不求损失梯度，而网络默认会求损失梯度
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()

    print(classes[int(predict)])#把索引index 传给classes，得到分类
    pre =torch.softmax(outputs,dim=1),#再次也可用softmax替代max来进行分类。
    print(pre)
if __name__ == '__main__':
    main()

