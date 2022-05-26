import torch
import torch.nn as nn
import copy

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, other_num_classes=0, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_class = num_classes

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.my_conv1 = nn.Conv2d(4, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        #self.my_conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.layer4_1 = copy.deepcopy(self.layer4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = nn.BatchNorm1d(512*block.expansion)
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes + other_num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            #### make last layer's stride equal 1       << just change this
            # if planes == 512:
            #     stride=1
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []


        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.my_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)





def _resnet(arch, block, layers, num_c, pretrained, progress, num_cc=0, **kwargs):
    model = ResNet(block, layers, num_classes = num_c, other_num_classes=num_cc, **kwargs)
    model.apply(weight_init)

    '''if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model_dict = model.state_dict()
        filtered_dict = {k:v for k, v in state_dict.items() if (k in model_dict and len(model_dict[k]) == len(v))}

        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)'''
    return model

def resnet50(num_c, num_cc=0, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], num_c, pretrained, progress, num_cc,
                   **kwargs)





from torch.utils.data import DataLoader
import data_manage

PATH = '/home/MMI22simchan/Mobticon/'
test_dataset = data_manage.CustomDataset(PATH+'test.csv',PATH,mode='test')
train_dataset = data_manage.CustomDataset(PATH+'Labeling_ver4_crop_finished.csv',PATH,mode='Train')

batch =32
testloader = DataLoader(test_dataset,batch_size=49,shuffle=True,num_workers=0)
trainloader = DataLoader(train_dataset,batch_size=batch,shuffle=True,num_workers=0)


import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
import torch.optim as optim
import os

def train(epochs):
    model = resnet50(num_c=5, pretrained=False)
    #model = torch.nn.DataParallel(model,device_ids=[0,1])
    start_epoch =0
    best_score = 0
    score = 0

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           epochs * batch,
                                                           eta_min=0.005 / 10)
    for epoch in range(epochs):
        OOD_data = 0  # for error control
        total_loss = 0

        class_running_loss = 0
        train_acc = 0
        train_label = [0] * 6
        test_acc = 0
        stime = time.time()
        for i,data in enumerate(trainloader,0):
            
            model = model.to(device)
            optimizer.zero_grad()

            image = data['image'].type(torch.FloatTensor)
            image = image.to(device)
      

            label = data['landmarks'].type(torch.LongTensor).to(device)
     
            output = model(image)
            
            class_loss = criterion(output, label)
            class_running_loss+=class_loss

            class_loss.backward()
            optimizer.step()
            scheduler.step()
        test_acc = 0
        test_label = [0] * 6
        with torch.no_grad():
            tot=0  
            cor=0
            model = model.to(device)
            for i, data in enumerate(testloader, 0):
            
                org_image = data['image'].type(torch.FloatTensor)
            
                org_image=org_image.to(device)
                gt = data['landmarks'].type(torch.FloatTensor)
              
                gt=gt.to(device)
                
                
                model = model.to(device)
               

                output = model(org_image)
                 
                
                #gt_label = torch.argmax(gt, dim=1).cpu().detach().tolist()
                
                
                output_label = torch.argmax(torch.sigmoid(output), dim=1).cpu().detach().tolist()
               
        
                
                for idx, label in enumerate(gt):
                    tot+=1
                    if label == output_label[idx]:
                
                        cor+=1
                        
        print('Epoch:{},Elapsed time:{:.2f},lr: {:.4f}*e-4'.format(epoch,time.time()-stime,scheduler.get_last_lr()[0] * 10 ** 4))
        print('class_loss = {:.10f}'.format(class_running_loss/batch))
        print('valid_err = {:.3f}'.format(cor/tot))
        
  
        if epoch == 20 or 40 or 60 or 80 or 99:

            torch.save(model.state_dict(),'/home/MMI22simchan/Mobticon/cp220512/checkpoint{}.pth'.format(epoch))




train(100)


