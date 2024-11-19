
import torch.nn as nn

import torch.utils.data

import torch.nn.functional as F


class baseconv(nn.Module):

    def __init__(self,inputchannel):
        super(baseconv, self).__init__()
        self.Conv1 = nn.Conv2d(inputchannel, int(inputchannel*1.5), kernel_size=3, stride=2)
        self.batchnorm=nn.BatchNorm2d(int(inputchannel*1.5))

        self.relu=nn.LeakyReLU()

    def forward(self, input):
        x=self.Conv1(input)
        x=self.batchnorm(x)
        x=self.relu(x)
        return x



class GazeCNN(nn.Module):
    def __init__(self):
        super(GazeCNN, self).__init__()
        self.baseConv1 = nn.Conv2d(1,16, kernel_size=3, stride=2)
        self.dropout=nn.Dropout(0.1)
        self.Conv1= baseconv(16)
        self.Conv2 = baseconv(24)
        self.Conv3 = baseconv(36)
        self.Conv4 = baseconv(54)
        self.Conv5 = baseconv(81)
        self.FC1=nn.AdaptiveAvgPool2d(1)


    def forward(self, input):
        x = self.baseConv1(input)
        x=F.relu(x)
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.dropout(x)
        x = self.Conv5(x)
        x =self.dropout(x)
        # print(x.shape)

        x=self.FC1(x)
        x=x.reshape(input.shape[0],121)

        return x


class NVgaze(nn.Module):
    def __init__(self):
        super(NVgaze, self).__init__()
        self.eye = GazeCNN()
        self.prediciton=nn.Linear(121,2)

    def forward(self,eye):
        x1 =  self.eye(eye)
        x= self.prediciton(x1)
        return x





if __name__ == '__main__':
    input = torch.zeros( 3, 1, 224, 224).cuda()
    model = NVgaze().cuda()
    j = model(input)
    print(j.shape)