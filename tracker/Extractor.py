import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

from .Net import ReID_Net

class Extractor(object):
    def __init__(self, model_path, device, num_classes=2915):

        # self.args = args
        self.net = ReID_Net(reid=True,num_classes=num_classes)
        self.device = device
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        print("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (32, 32)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):

        def _resize(im, size):
            # print("im.shape", im.shape)
            return cv2.resize(im.astype(np.float32) / 255., size)
        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()

        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
            # print(features.shape #[3,256]
        return features.cpu().numpy()


if __name__ == '__main__':
    extractor = Extractor("path/to/weights.pth", device="cuda:3")
    print("success!")