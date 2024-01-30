import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append("../")

from nets.backbone import Backbone, C2f, Conv
from nets.yolo_training import weights_init
from nets.swin import Swin
from nets.convnext import ConvNeXt
from utils.utils_bbox import make_anchors

def fuse_conv_and_bn(conv, bn):
    # 混合Conv2d + BatchNorm2d 减少计算量
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备kernel
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

class DFL(nn.Module):
    """
    DFL模块
    Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391 https://arxiv.org/abs/2006.04388
    使用卷积的操作进行相乘和相加

    预测结果取softmax   0.0	 0.1  0.0  0.0  0.4  0.5  0.0  0.0
    参考的固定值         0    1    2    3    4    5    6    7   conv中的权重
    点乘               0.1 * 1 + 0.4 * 4 + 0.5 * 5 = 4.2
    """
    def __init__(self, c1=16):
        super().__init__()
        #-----------------------------------------------------#
        #   不训练这个conv,权重设置为 [0...15], 直接和预测值相乘
        #   out_channels = 1,含义为计算结果channel为1
        #-----------------------------------------------------#
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x           = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))    # 权重设置为 [0...15]
        self.c1     = c1

    def forward(self, x):
        # [B, 4 * c1, 8400]
        b, c, a = x.shape     # view                transpose           softmax             conv               view
        # [B, 4 * c1, 8400] => [B, 4, c1, 8400] => [B, c1, 4, 8400] => [B, c1, 4, 8400] => [B, 1, 4, 8400] => [B, 4, 8400]
        # 以softmax的方式，对0~16的数字计算百分比，获得最终数字。
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, input_shape, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        self.phi = phi
        if phi in ['n', 's', 'm', 'l', 'x']:
            depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
            width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
            deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
            dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

            base_channels       = int(wid_mul * 64)          # 64
            base_depth          = max(round(dep_mul * 3), 1) # 3
            #-----------------------------------------------#
            #   输入图片是3, 640, 640
            #-----------------------------------------------#

            #-----------------------------------------------#
            #   生成主干模型
            #   获得三个有效特征层，他们的shape分别是：
            #   [B, 256, 80, 80]
            #   [B, 512, 40, 40]
            #   [B, 1024 * deep_mul, 20, 20]
            #-----------------------------------------------#
            self.backbone   = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)

        elif 'swin' in phi or 'convnext' in phi:
            dep_mul, wid_mul, deep_mul = 1.00, 1.25, 0.50 # 使用x大小的参数

            base_channels   = int(wid_mul * 64)          # 64
            base_depth      = max(round(dep_mul * 3), 1) # 3
            #-----------------------------------------------#
            #   输入图片是3, 640, 640
            #-----------------------------------------------#
            if 'swin' in phi:
                self.backbone = Swin(variance=phi, pretrained=pretrained)
            elif 'convnext' in phi:
                self.backbone = ConvNeXt(variance=phi, pretrained=pretrained)
            feat1_in        = {'swin_t' : 192, 'swin_s' : 192, 'swin_b' : 256,  'swin_v2_t' : 192, 'swin_v2_s' : 192, 'swin_v2_b' : 256,
                               'convnext_tiny' : 192, 'convnext_small' : 192, 'convnext_base' : 256,  'convnext_large' : 384,}[phi]
            feat2_in        = {'swin_t' : 384, 'swin_s' : 384, 'swin_b' : 512,  'swin_v2_t' : 384, 'swin_v2_s' : 384, 'swin_v2_b' : 512,
                               'convnext_tiny' : 384, 'convnext_small' : 384, 'convnext_base' : 512,  'convnext_large' : 768,}[phi]
            feat3_in        = {'swin_t' : 768, 'swin_s' : 768, 'swin_b' : 1024, 'swin_v2_t' : 768, 'swin_v2_s' : 768, 'swin_v2_b' : 1024,
                               'convnext_tiny' : 768, 'convnext_small' : 768, 'convnext_base' : 1024, 'convnext_large' : 1536,}[phi]
            # 调整输出channel到符合PANet的输入channel
            self.for_feat1  = nn.Conv2d(feat1_in, base_channels * 4,  1)
            self.for_feat2  = nn.Conv2d(feat2_in, base_channels * 8,  1)
            self.for_feat3  = nn.Conv2d(feat3_in, int(base_channels * 16 * deep_mul), 1)

        #------------------------加强特征提取网络------------------------#
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        # [B, 1024 * deep_mul] + [B, 512, 40, 40] => [B, 512, 40, 40]
        self.conv3_for_upsample1    = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        # [B, 768, 80, 80] => [B, 256, 80, 80]
        self.conv3_for_upsample2    = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth, shortcut=False)

        # [B, 256, 80, 80] => [B, 256, 40, 40]
        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        # [B, 512 + 256, 40, 40] => [B, 512, 40, 40]
        self.conv3_for_downsample1  = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth, shortcut=False)

        # [B, 512, 40, 40] => [B, 512, 20, 20]
        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        # [B, 1024 * deep_mul] + [B, 512, 20, 20] =>  [B, 1024 * deep_mul, 20, 20]
        self.conv3_for_downsample2  = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
        #------------------------加强特征提取网络------------------------#

        ch              = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape      = None
        self.nl         = len(ch)

        # stride: [ 8., 16., 32.]
        # self.stride     = torch.zeros(self.nl)
        self.stride     = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward

        self.reg_max    = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no         = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.num_classes = num_classes

        c2, c3   = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
        # box => [B, 4 * reg_max, 80/40/20, 80/40/20]
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        # cls => [B, num_classes, 80/40/20, 80/40/20]
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        if not pretrained:
            weights_init(self)
        # box [B, 4 * reg_max, 8400] => [B, 4, 8400]
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self

    def forward(self, x):
        #---------------------------------------------------#
        #   backbone
        #   获得三个有效特征层，他们的shape分别是：
        #   feat1: [B, 256, 80, 80]
        #   feat2: [B, 512, 40, 40]
        #   feat3: [B, 1024 * deep_mul, 20, 20]
        #---------------------------------------------------#
        feat1, feat2, feat3 = self.backbone(x)
        if self.phi not in ['n', 's', 'm', 'l', 'x']:
            feat1 = self.for_feat1(feat1)
            feat2 = self.for_feat2(feat2)
            feat3 = self.for_feat3(feat3)

        #------------------------加强特征提取网络------------------------#
        # [B, 1024 * deep_mul, 20, 20] => [B, 1024 * deep_mul, 40, 40]
        P5_upsample = self.upsample(feat3)
        # [B, 1024 * deep_mul, 40, 40] cat [B, 512, 40, 40] => [B, 1024 * deep_mul + 512, 40, 40]
        P4          = torch.cat([P5_upsample, feat2], 1)
        # [B, 1024 * deep_mul + [512, 40, 40] =>[B, 512, 40, 40]
        P4          = self.conv3_for_upsample1(P4)

        # [B, 512, 40, 40] => [B, 512, 80, 80]
        P4_upsample = self.upsample(P4)
        # [B, 512, 80, 80] cat [B, 256, 80, 80] => [B, 768, 80, 80]
        P3          = torch.cat([P4_upsample, feat1], 1)
        # [B, 768, 80, 80] => [B, 256, 80, 80]
        P3_out      = self.conv3_for_upsample2(P3)

        # [B, 256, 80, 80] => [B, 256, 40, 40]
        P3_downsample = self.down_sample1(P3_out)
        # [B, 512, 40, 40] cat [B, 256, 40, 40] => [B, 768, 40, 40]
        P4 = torch.cat([P3_downsample, P4], 1)
        # [B, 768, 40, 40] => [B, 512, 40, 40]
        P4_out = self.conv3_for_downsample1(P4)

        # [B, 512, 40, 40] => [B, 512, 20, 20]
        P4_downsample = self.down_sample2(P4_out)
        # [B, 512, 20, 20] cat [B, 1024 * deep_mul, 20, 20] => [B, 1024 * deep_mul + 512, 20, 20]
        P5 = torch.cat([P4_downsample, feat3], 1)
        # [B, 1024 * deep_mul + 512, 20, 20] => [B, 1024 * deep_mul, 20, 20]
        P5_out = self.conv3_for_downsample2(P5)
        #------------------------加强特征提取网络------------------------#
        # P3_out [B, 256, 80, 80]
        # P4_out [B, 512, 40, 40]
        # P5_out [B, 1024 * deep_mul, 20, 20]
        shape = P3_out.shape  # BCHW

        # 将每层的box和cls输出拼接起来
        # P3_out [B, 256, 80, 80]             => [B, 4 * reg_max + num_classes, 80, 80]
        # P4_out [B, 512, 40, 40]             => [B, 4 * reg_max + num_classes, 40, 40]
        # P5_out [B, 1024 * deep_mul, 20, 20] => [B, 4 * reg_max + num_classes, 20, 20]
        x = [P3_out, P4_out, P5_out]
        for i in range(self.nl):
            # cv2: box => [B, 4 * reg_max, 80, 80]
            # cv3: cls => [B, num_classes, 80, 80]
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.shape != shape:
            # anchors: [8400, 2] -> [2, 8400]
            # strides: [8400, 1] -> [1, 8400]
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # 将不同层的box和cls分别拼接到一起
        # [B, 4 * reg_max + num_classes, 80, 80] => [B, 4 * reg_max + num_classes, 80*80] ───┐
        # [B, 4 * reg_max + num_classes, 40, 40] => [B, 4 * reg_max + num_classes, 40*40] ─ cat => [B, 4 * reg_max + num_classes, 8400]
        # [B, 4 * reg_max + num_classes, 20, 20] => [B, 4 * reg_max + num_classes, 20*20] ───┘
        # 依次取出每层的输出,然后在最后维度拼接,之后在第1个维度分离出box和cls
        # [B, 4 * reg_max + num_classes, 8400] split [B, 4 * reg_max, 8400] : box
        #                                            [B, num_classes, 8400] : cls
        box, cls        = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.num_classes), 1)
        # origin_cls      = [xi.split((self.reg_max * 4, self.num_classes), 1)[1] for xi in x]

        # 对box做DFL处理,获取位置的4个回归值
        # [B, 4 * reg_max, 8400] => [B, 4, 8400]
        dbox            = self.dfl(box)

        #-------------------------------------------------------------------------------------------#
        #   dbox:    [B, 4, 8400]           box detect
        #   cls:     [B, num_classes, 8400] cls detect
        #   x:       [[B, 144, 80, 80], [B, 144, 40, 40], [B, 144, 20, 20]]   [P3_out, P4_out, P5_out]
        #   anchors: [2, 8400]
        #   strides: [1, 8400]
        #-------------------------------------------------------------------------------------------#
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)


if __name__ == "__main__":
    phi = "x"
    model = YoloBody(input_shape=[640, 640], num_classes=80, phi=phi, pretrained=False)
    x = torch.ones(4, 3, 640, 640)

    model.eval()
    with torch.inference_mode():
        dbox, cls, x, anchors, strides = model(x)
    print(dbox.shape)     # [B, 4, 8400]
    print(cls.shape)      # [B, 80, 8400]
    print(x[0].shape)     # [B, 144, 80, 80]  144 = 4 * reg_max + num_classes = 16 * 4 + 80
    print(x[1].shape)     # [B, 144, 40, 40]
    print(x[2].shape)     # [B, 144, 20, 20]
    print(anchors.shape)  # [2, 8400]
    print(strides.shape)  # [1, 8400]
    print(anchors[:2, :5])
    # [[0.5000, 1.5000, 2.5000, 3.5000, 4.5000]
    #  [0.5000, 0.5000, 0.5000, 0.5000, 0.5000]]
    print(strides[:, :5])
    # [[8., 8., 8., 8., 8.]]

    if False:
        onnx_path = f"yolov8{phi}.onnx"
        torch.onnx.export(
            model=model,
            args=x,
            f=onnx_path,
            input_names=['image'],
        )
        import onnx
        from onnxsim import simplify

        # 载入onnx模型
        model_ = onnx.load(onnx_path)

        # 简化模型
        model_simple, check = simplify(model_)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simple, onnx_path)
        print('finished exporting ' + onnx_path)
