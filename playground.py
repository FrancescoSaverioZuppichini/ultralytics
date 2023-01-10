from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
from torch import nn
import torch

print(model.model)

# model.train(data="coco128.yaml")  # train the model
# model.val()  # evaluate model performance on the validation set
model.predict(source="https://ultralytics.com/images/bus.jpg")  # predict on an image
# model.export(format="onnx")  # export the model to ONNX format


from ultralytics.nn.modules import C2f

cf2 = C2f(32, 32, n=2)
cf2(torch.randn((1, 32, 8, 8)))


# yolov5 shit

# class CBL(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super(CBL, self).__init__()

#         conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)

#         self.cbl = nn.Sequential(
#             conv,
#             bn,
#             # https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
#             nn.SiLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.cbl(x)


# class Bottleneck(nn.Module):
#     """
#     Parameters:
#         in_channels (int): number of channel of the input tensor
#         out_channels (int): number of channel of the output tensor
#         width_multiple (float): it controls the number of channels (and weights)
#                                 of all the convolutions beside the
#                                 first and last one. If closer to 0,
#                                 the simpler the modelIf closer to 1,
#                                 the model becomes more complex
#     """
#     def __init__(self, in_channels, out_channels, width_multiple=1):
#         super(Bottleneck, self).__init__()
#         c_ = int(width_multiple*in_channels)
#         self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
#         self.c2 = CBL(c_, out_channels, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         return self.c2(self.c1(x)) + x

# class C3(nn.Module):
#     """
#     Parameters:
#         in_channels (int): number of channel of the input tensor
#         out_channels (int): number of channel of the output tensor
#         width_multiple (float): it controls the number of channels (and weights)
#                                 of all the convolutions beside the
#                                 first and last one. If closer to 0,
#                                 the simpler the modelIf closer to 1,
#                                 the model becomes more complex
#         depth (int): it controls the number of times the bottleneck (residual block)
#                         is repeated within the C3 block
#         backbone (bool): if True, self.seq will be composed by bottlenecks 1, if False
#                             it will be composed by bottlenecks 2 (check in the image linked below)
#         https://user-images.githubusercontent.com/31005897/172404576-c260dcf9-76bb-4bc8-b6a9-f2d987792583.png
#     """
#     def __init__(self, in_channels, out_channels, width_multiple=1, depth=1, backbone=True):
#         super(C3, self).__init__()
#         c_ = int(width_multiple*in_channels)

#         self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
#         self.c_skipped = CBL(in_channels,  c_, kernel_size=1, stride=1, padding=0)
#         if backbone:
#             self.seq = nn.Sequential(
#                 *[Bottleneck(c_, c_, width_multiple=1) for _ in range(depth)]
#             )
#         else:
#             self.seq = nn.Sequential(
#                 *[nn.Sequential(
#                     CBL(c_, c_, 1, 1, 0),
#                     CBL(c_, c_, 3, 1, 1)
#                 ) for _ in range(depth)]
#             )
#         self.c_out = CBL(c_ * 2, out_channels, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         x = torch.cat([self.seq(self.c1(x)), self.c_skipped(x)], dim=1)
#         return self.c_out(x)


# print(C3(32, 32))
