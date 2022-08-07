import numpy as np
import torch
import cv2
import os
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import time
import argparse
from models.BANet_Plus_model import BANet_Plus_model

def get_args():
	parser = argparse.ArgumentParser('Test an image')
	parser.add_argument('--weights_path', required=True, help='Weights path')
	return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    blur_path = './datasets/HIDE/blur/'
    out_path = './out/BANet_Plus_HIDE_result'
    model = nn.DataParallel(BANet_Plus_model())
    model = model.cuda()
    model.load_state_dict(torch.load(args.weights_path))
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    test_time = 0
    iteration = 0
    total_image_number = 1111

    # warm up
    warm_up = 0
    for file in os.listdir(blur_path):
        if not os.path.isdir(out_path + '/' + file):
            os.mkdir(out_path + '/' + file)
        for img_name in os.listdir(blur_path + '/' + file):
            warm_up += 1
            img = cv2.imread(blur_path + '/' + file + '/' + img_name)
            img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32')) - 0.5
            with torch.no_grad():
                img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()
                result_image = model(img_tensor)
            if warm_up == 10:
                break
        break

    for file in os.listdir(blur_path):
        if not os.path.isdir(out_path + '/' + file):
            os.mkdir(out_path + '/' + file)
        for img_name in os.listdir(blur_path + '/' + file):
            img = cv2.imread(blur_path + '/' + file + '/' + img_name)
            img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32')) - 0.5
            with torch.no_grad():
                iteration += 1
                img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()

                start = time.time()
                result_image = model(img_tensor).clamp(-0.5, 0.5)
                stop = time.time()

                print('Image:{}/{}, CNN Runtime:{:.4f}'.format(iteration, total_image_number, (stop - start)))
                test_time += stop - start
                print('Average Runtime:{:.4f}'.format(test_time / float(iteration)))
                result_image = result_image + 0.5
                out_file_name = out_path + '/' + file + '/' + img_name
                torchvision.utils.save_image(result_image[:, [2, 1, 0]], out_file_name)
