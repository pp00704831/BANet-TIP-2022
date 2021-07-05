from __future__ import print_function
import argparse
import numpy as np
import torch
import cv2
import yaml
import os
from torchvision import models, transforms
from torch.autograd import Variable
import glob
import tqdm
from util.metrics import PSNR
from models.networks import get_generator
import torchvision
import time

def get_args():
	parser = argparse.ArgumentParser('Test an image')
	parser.add_argument('--weights_path', required=True, help='Weights path')
	return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    with open('config/config.yaml') as cfg:
        config = yaml.load(cfg)
    model = get_generator(config['model'])
    model.load_state_dict(torch.load(args.weights_path))
    model = model.cuda()
    blur_path = './datasets/GoPro/test/blur/'
    out_path = './out/BANet_GoPro_result'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    test_time = 0
    iteration = 0
    total_image_number = 1111
    for file in os.listdir(blur_path):
        if not os.path.isdir(out_path + '/' + file):
            os.mkdir(out_path + '/' + file)
        for img_name in os.listdir(blur_path + '/' + file):
            img = cv2.imread(blur_path + '/' + file + '/' + img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32')) - 0.5
            with torch.no_grad():
                iteration += 1
                img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()

                start = time.time()
                result_image = model(img_tensor)
                stop = time.time()

                print('Image:{}/{}, CNN Runtime:{:.4f}'.format(iteration, total_image_number, (stop - start)))
                test_time += stop - start
                print('Average Runtime:{:.4f}'.format(test_time / float(iteration)))
                result_image = result_image + 0.5
                out_file_name = out_path + '/' + file + '/' + img_name
                torchvision.utils.save_image(result_image, out_file_name)
