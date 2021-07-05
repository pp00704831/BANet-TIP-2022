# BANet
Pytorch Implementation of "[BANet: Blur-aware Attention Networks for Dynamic Scene Deblurring](https://arxiv.org/abs/2101.07518)"


<img src="./figure/Disentangle.png" width = "1000" height = "400" div align=center />
<img src="./figure/Architecture.png" width = "1000" height = "250" div align=center />


## Installation
```
git clone https://github.com/pp00704831/BANet.git
cd BANet
Conda create -n banet python=3.6
source activate banet
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install opencv-python tqdm pyyaml joblib glog scikit-image tensorboardX albumentations==1.0.0
```

### **Training**
Download "[GoPro](https://drive.google.com/drive/folders/1sZokl0e1NIbQE9DF5d4q75nYTcX7nHvk?usp=sharing)" dataset into './datasets' </br>
For example: 
'./datasets/GoPro/train/blur/\*\*/\*.png'

**For training, run following commands.**
```
python train.py
```
### **Testing**
For reproducing our results, download the GoPro trained model "[BANet_GoPro.pth](https://drive.google.com/drive/folders/1sZokl0e1NIbQE9DF5d4q75nYTcX7nHvk?usp=sharing)" into './checkpoints'

**For testing on GoPro dataset**
```
python predict_BANet_GoPro_test_results.py --weights_path ./checkpoints/BANet_GoPro.pth 
```
**For testing on HIDE dataset** </br>
Download HIDE dataset into './datasets'
```
python predict_BANet_HIDE_test_results.py --weights_path ./checkpoints/BANet_GoPro.pth 
```
**For testing your own training weight on GoPro or HIDE**  </br>

Take GoPro for example
* Chang the 'output_path' in 'predict_BANet_GoPro_test_results.py'
* Chage weight path command to --weights_path ./final_BANet.pth 


### **Evaluation**
**For evaluation on GoPro results in MATLAB**
Download "[BANet_GoPro_result](https://drive.google.com/drive/folders/1sZokl0e1NIbQE9DF5d4q75nYTcX7nHvk?usp=sharing)" into './out'
```
evaluation_GoPro.m
```
**For evaluation on HIDE results in MATLAB**
Download "[BANet_HIDE_result](https://drive.google.com/drive/folders/1sZokl0e1NIbQE9DF5d4q75nYTcX7nHvk?usp=sharing)" into './out'
```
evaluation_HIDE.m
```
