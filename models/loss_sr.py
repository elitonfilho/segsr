import torch
from torch import nn
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    def __init__(self, seg='hrnet'):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.seg_loss = getSegLoss(seg)

    def forward(self, out_labels, out_images, target_images, seg_label=None, seg_pred=None, use_seg=True):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        seg_loss = self.seg_loss(seg_pred, seg_label) if use_seg else 0
        losses = {
            "image_loss": image_loss.item(),
            "adversarial_loss": adversarial_loss.item(),
            "perception_loss":perception_loss.item(),
            "tv_loss": tv_loss.item(),
            "seg_loss": seg_loss.item()
        }
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss + 0.001 * seg_loss, losses


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # weights = torch.tensor([.1, .3, .3, .3])
        # self.CEE = nn.CrossEntropyLoss(weight=weights)
        self.CEE = nn.CrossEntropyLoss()

    def forward(self, pred, label):
        return self.CEE(pred, label)

    # def forward(self, label, pred):
    #     t_sum, t_acc = 0, 0
    #     for i in range(pred.shape[0]):
    #         valid = (label[i,0] >= 0)
    #         acc_sum = (valid * (pred[i,3] == label[i,0])).sum()
    #         # print(valid.shape, acc_sum)
    #         valid_sum = valid.sum()
    #         acc = float(acc_sum) / (valid_sum + 1e-10)
    #         t_sum += valid_sum
    #         t_acc += acc
    #     return t_acc


def getSegLoss(loss):
    if loss == 'hrnet':
        return SegLoss()
    elif loss == 'unet':
        return SegLoss()
    else:
        return None


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
