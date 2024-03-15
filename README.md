# BANet (TIP 2022)
Pytorch Implementation of "[BANet: Blur-aware Attention Networks for Dynamic Scene Deblurring](https://arxiv.org/abs/2101.07518)"


<img src="./figure/Architecture.png" width = "1000" height = "250" div align=center />


## Installation
The implementation of our BANet is modified from "[DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2)"
```
git clone https://github.com/pp00704831/BANet.git
cd BANet
conda create -n banet python=3.6
source activate banet
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install opencv-python tqdm pyyaml joblib glog scikit-image tensorboardX albumentations
pip install -U albumentations[imgaug]
pip install albumentations==1.1.0
```

### **Training**
Download "[GoPro](https://drive.google.com/drive/u/1/folders/0B1M5aNe46QPlflAtXzR2RzhwQWkzbGoyaS1HYlVSY2FBd1NMZ1BHbFFEdnROQl9NQjZDNVE?resourcekey=0-MBS_f0fXHrFCK2EiMswB6Q)" dataset into './datasets' </br>
For example: 
'./datasets/GoPro/train/blur/\*\*/\*.png'

**We train our BANner in two stages:** </br>
**1) We pre-train for 3000 epochs on patch size 256x256. Please run the following commands.** </br>
```
python pretrained.py
```

**2) We fine-tune for 1000 epochs on patch size 512x512. Please run the following commands.** </br>
```
python train.py
```

### **Testing**
Download "[GoPro](https://drive.google.com/drive/u/1/folders/0B1M5aNe46QPlflAtXzR2RzhwQWkzbGoyaS1HYlVSY2FBd1NMZ1BHbFFEdnROQl9NQjZDNVE?resourcekey=0-MBS_f0fXHrFCK2EiMswB6Q)" dataset into './datasets' </br>
For example: 
'./datasets/GoPro/test/blur/\*\*/\*.png' </br>
For reproducing our results, download the GoPro trained model "[BANet_GoPro.pth](https://drive.google.com/drive/u/1/folders/0B1M5aNe46QPlflAtXzR2RzhwQWkzbGoyaS1HYlVSY2FBd1NMZ1BHbFFEdnROQl9NQjZDNVE?resourcekey=0-MBS_f0fXHrFCK2EiMswB6Q)

**For testing on GoPro dataset**
```
python predict_BANet_GoPro_test_results.py --weights_path ./BANet_GoPro.pth 
```
**For testing on HIDE dataset** </br>
Download "[HIDE](https://drive.google.com/drive/u/1/folders/0B1M5aNe46QPlflAtXzR2RzhwQWkzbGoyaS1HYlVSY2FBd1NMZ1BHbFFEdnROQl9NQjZDNVE?resourcekey=0-MBS_f0fXHrFCK2EiMswB6Q)" dataset into './datasets' </br>
```
python predict_BANet_HIDE_test_results.py --weights_path ./BANet_GoPro.pth 
```
**For testing your own training weight on GoPro or HIDE**  </br>

Take GoPro for example
* Rename the 'output_path' in line 23 in the predict_BANet_GoPro_test_results.py
* Chage weight path command to --weights_path ./final_BANet_GoPro.pth 


### **Evaluation**
**For evaluation on GoPro results in MATLAB**
Download "[BANet_GoPro_result](https://drive.google.com/drive/u/1/folders/0B1M5aNe46QPlflAtXzR2RzhwQWkzbGoyaS1HYlVSY2FBd1NMZ1BHbFFEdnROQl9NQjZDNVE?resourcekey=0-MBS_f0fXHrFCK2EiMswB6Q)" into './out'
```
evaluation_GoPro.m
```
**For evaluation on HIDE results in MATLAB**
Download "[BANet_HIDE_result](https://drive.google.com/drive/u/1/folders/0B1M5aNe46QPlflAtXzR2RzhwQWkzbGoyaS1HYlVSY2FBd1NMZ1BHbFFEdnROQl9NQjZDNVE?resourcekey=0-MBS_f0fXHrFCK2EiMswB6Q)" into './out'
```
evaluation_HIDE.m
```

### **BANet+** 
The training strategy is the same as BANet above. </br>
### **Testing**
Download the GoPro trained model "[BANet_Plus_GoPro.pth](https://drive.google.com/drive/u/1/folders/0B1M5aNe46QPlflAtXzR2RzhwQWkzbGoyaS1HYlVSY2FBd1NMZ1BHbFFEdnROQl9NQjZDNVE?resourcekey=0-MBS_f0fXHrFCK2EiMswB6Q) </br>
**For testing on GoPro dataset**
```
python predict_BANet_Plus_GoPro_test_results.py --weights_path ./BANet_Plus_GoPro.pth 
```
**For testing on HIDE dataset**
```
python predict_BANet_Plus_HIDE_test_results.py --weights_path ./BANet_Plus_GoPro.pth 
```
