import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.distributions.kl import kl_divergence as KLD
from torch.nn.functional import softplus
from models.mynn import initialize_weights, Norm2d
import models.GatedSpatialConv as gsc
import models.Resnet as Resnet
from models.wider_resnet import WiderResNetA2

import cv2
import numpy as np

device = torch.device('cuda:0')
class _AtrousSpatialPyramidPoolingModule(nn.Module):

    # operations performed:
    # 1x1 x depth
    # 3x3 x depth dilation 6
    # 3x3 x depth dilation 12
    # 3x3 x depth dilation 18
    # image pooling
    # concatenate all together
    # Final 1x1 conv

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()
        # _AtrousSpatialPyramidPoolingModule(4096, 256,output_stride=8)
        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x, edge):
        x_size = x.size()
        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        out = img_features
        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear', align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)
        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out
class convBlock(nn.Module):
    def __init__(self, inCh, nhid, nOp, ker=3,padding=1):
        super(convBlock,self).__init__()

        self.enc1 = nn.Conv2d(inCh,nhid,kernel_size=ker,padding=padding)
        self.enc2 = nn.Conv2d(nhid,nOp,kernel_size=ker,padding=padding)
        self.bn = nn.BatchNorm2d(inCh)
        self.act=nn.ReLU()
        self.scale=nn.Upsample(scale_factor=2)

    def forward(self,x):
        x = self.scale(x)
        x = self.bn(x)
        #x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=alin)
        x = self.act(self.enc1(x))
        x = self.act(self.enc2(x))
        return x
class inceptionBlock(nn.Module):
    def __init__(self, inCh, nhid,dilasize=2,dilapadding=1):
        super(inceptionBlock,self).__init__()

        self.bn=nn.BatchNorm2d(inCh)
        self.act=nn.ReLU()

        self.oneCov=nn.Conv2d(inCh,int(nhid*0.125),kernel_size=1)

        #self.threeConv1=nn.Conv2d(inCh,int(nhid*0.125),kernel_size=1)
        self.threeConv2=nn.Conv2d(inCh,int(nhid*0.125),kernel_size=3,padding=1,stride=2)
        #self.threeConv1x3=nn.Conv2d(inCh,int(nhid*0.125),kernel_size=(1,3),padding=(0,1))
        #self.threeConv3x1=nn.Conv2d(int(nhid*0.125),int(nhid*0.125),kernel_size=(3,1),padding=(1,0))

        #self.fiveConv1 = nn.Conv2d(inCh,int(nhid*0.75),kernel_size=1)
        self.fiveConv2 = nn.Conv2d(inCh,int(nhid*0.625),kernel_size=3,padding=1,stride=2)
        self.fiveConv3 = nn.Conv2d(int(nhid*0.625),int(nhid*0.625), kernel_size=3, padding=1)

        #self.dilaconv1=nn.Conv2d(inCh,int(nhid*0.125),kernel_size=1)
        #self.dila1 = nn.Conv2d(inCh, int(nhid * 0.125), kernel_size=3, stride=2, padding=1)
        self.dila = nn.Conv2d(inCh,int(nhid*0.125),kernel_size=3,stride=2,padding=dilapadding,dilation=dilasize)
        #self.dila2=nn.Conv2d(inCh,int(nhid*0.125),kernel_size=3,stride=2,padding=4,dilation=4)
        self.rrr = nn.Conv2d(inCh, nhid, kernel_size=3, padding=1)

        self.scale=nn.AvgPool2d(kernel_size=2)


    def forward(self,x):
        x=self.bn(x)
        #x=self.scale(x)
        x1=self.act(self.oneCov(self.scale(x)))
        x3=self.act(self.threeConv2(x))
        x41=self.act(self.dila(x))
        #x42=self.act(self.dila2(x))
        x5=self.act(self.fiveConv3(self.act(self.fiveConv2(x))))
        x5=torch.cat((x1,x3,x41,x5),dim=1)
        x=self.rrr(self.scale(x))
        x5+=x
        return x5

class U_net(nn.Module):
    def __init__(self, nhid=8, ker=3, inCh=1):
        super(U_net, self).__init__()
        #Encoder
        self.uEnc11=nn.Conv2d(inCh,nhid,kernel_size=ker,padding=1)
        self.uEnc12=nn.Conv2d(nhid,nhid,kernel_size=ker,padding=1)

        self.uEnc2=inceptionBlock(nhid,2*nhid,dilasize=2,dilapadding=2)
        self.uEnc3=inceptionBlock(2*nhid,4*nhid,dilasize=2,dilapadding=2)
        self.uEnc4=inceptionBlock(4*nhid,8*nhid,dilasize=2,dilapadding=2)
        self.uEnc5=inceptionBlock(8*nhid,16*nhid,dilapadding=2,dilasize=2)

        self.act=nn.ReLU()
        #self.bn=nn.BatchNorm2d(inCh)

        self.dsn1 = nn.Conv2d(nhid, 1, 1)
        self.dsn3 = nn.Conv2d(4 * nhid, 1, 1)
        self.dsn4 = nn.Conv2d(8 * nhid, 1, 1)
        self.dsn5 = nn.Conv2d(16 * nhid, 1, 1)

        self.res1 = Resnet.BasicBlock(nhid, nhid, stride=1, downsample=None)
        self.d1 = nn.Conv2d(nhid, int(nhid / 2), 1)
        self.res2 = Resnet.BasicBlock(int(nhid / 2), int(nhid / 2), stride=1, downsample=None)
        self.d2 = nn.Conv2d(int(nhid / 2), int(nhid / 4), 1)
        self.res3 = Resnet.BasicBlock(int(nhid / 4), int(nhid / 4), stride=1, downsample=None)
        self.d3 = nn.Conv2d(int(nhid / 4), int(nhid / 8), 1)
        self.fuse = nn.Conv2d(int(nhid / 8), 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(int(nhid / 2), int(nhid / 2))
        self.gate2 = gsc.GatedSpatialConv2d(int(nhid / 4), int(nhid / 4))
        self.gate3 = gsc.GatedSpatialConv2d(int(nhid / 8), int(nhid / 8))

        self.aspp = _AtrousSpatialPyramidPoolingModule(16*nhid, nhid,
                                                       output_stride=8)

        self.sigmoid = nn.Sigmoid()
        #initialize_weights(self.final_seg)


        #Decoder

        self.dec5 = convBlock(16 * nhid + 6 * nhid, 8 * nhid,8*nhid)
        # self.att4 = Attention_block(8*nhid,8*nhid,8*nhid)
        self.dec4 = convBlock(16 * nhid, 4 * nhid,4*nhid)
        # self.att3 = Attention_block(4 * nhid, 4 * nhid, 4 * nhid)
        self.dec3 = convBlock(8 * nhid, 2 * nhid,2*nhid)
        # self.att2 = Attention_block(2 * nhid, 2 * nhid, 2 * nhid)
        self.dec2 = convBlock(4 * nhid, nhid,nhid)
        # self.att1 = Attention_block(nhid, nhid, nhid)
        self.dec11 = nn.Conv2d(2 * nhid, nhid, kernel_size=ker, padding=1)
        self.dec12 = nn.Conv2d(nhid, inCh, kernel_size=ker, padding=1)



    def uEncoder(self,input):
        x_size = input.size()
        x=[]

        x.append(self.act(self.uEnc12(self.act(self.uEnc11(input)))))#16

        x.append(self.uEnc2(x[0]))#32

        x.append(self.uEnc3(x[1]))#64

        x.append(self.uEnc4(x[2]))#128

        x.append(self.uEnc5(x[3]))#256

        s3 = F.interpolate(self.dsn3(x[2]), x_size[2:],
                           mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.dsn4(x[3]), x_size[2:],
                           mode='bilinear', align_corners=True)
        s5 = F.interpolate(self.dsn5(x[4]), x_size[2:],
                           mode='bilinear', align_corners=True)

        m1f = F.interpolate(x[0], x_size[2:], mode='bilinear', align_corners=True)

        im_arr = input.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()

        cs = self.res1(m1f)  # 16
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d1(cs)  # 8
        cs = self.gate1(cs, s3)  # 8
        cs = self.res2(cs)  # 8
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d2(cs)  # 4
        cs = self.gate2(cs, s4)  # 4
        cs = self.res3(cs)  # 4
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d3(cs)  # 2
        cs = self.gate3(cs, s5)  # 2
        cs = self.fuse(cs)  # 1
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)

        edge_out = self.sigmoid(cs)#1
        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)

        xx = self.aspp(x[4], acts)#96

        '''dec0_up = self.bot_aspp(xx)

        dec0_fine = self.bot_fine(x2)
        dec0_up = F.interpolate(dec0_up, x2.size()[2:], mode='bilinear', align_corners=True)
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        dec1 = self.final_seg(dec0)
        seg_out = F.interpolate(dec1, x_size[2:], mode='bilinear')'''

        return x,xx,edge_out
    def uDecoder(self,x_enc,xx):
        x=torch.cat((x_enc[-1],xx),dim=1)#352
        x = self.dec5(x)  # 8
        x = torch.cat((x, x_enc[-2]), dim=1)  # 16

        x = self.dec4(x)  # 4

        x = torch.cat((x, x_enc[-3]), dim=1)  # 8
        x = self.dec3(x)  # 2

        x = torch.cat((x, x_enc[-4]), dim=1)  # 4
        x = self.dec2(x)  # 1

        x = torch.cat((x, x_enc[-5]), dim=1)
        x = self.act(self.dec11(x))
        x = self.dec12(x)

        return x
    def forward(self, x):

        x_enc,xx,edg = self.uEncoder(x)


        xHat = self.uDecoder(x_enc,xx)
        return xHat,edg