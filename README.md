# ForkNet
An end-to-end convolutional neural network for the DoFP sensor to reconstruct three common polarization properties, which is implemented with Tensorflow.

## Network Architecture
![ForkNet](https://github.com/AGroupofProbiotocs/ForkNet/blob/master/ForkNet.jpg)  

## Notes
1. The input patches and labels can be generated by running the "generate_labels" file in the "utils" folder if your have polarization intensity images at four orientations in hand. 
2. The original model checkpoint which achieves the PSNR showed on the paper has been unfortunately lost due to file movement. But we has temporarily retrained a new model which shows a similar performance as the previous one and saved it in the "best_model" folder. 
3. Our dataset of the raw intensity images at four orientations can be obtained via the following link: [Raw Data on Onedrive](https://emailszueducn-my.sharepoint.com/:f:/g/personal/2172281561_email_szu_edu_cn/EurMDr8smshFs6Sk2wkBS2UBjPuskpfKUP6ylOxqvJKDXw?e=UP2pRM)

## Citation
Please cite the following paper if you use the code:

[Xianglong Zeng, Yuan Luo, Xiaojing Zhao, and Wenbin Ye, "An end-to-end fully-convolutional neural network for division of focal plane sensors to reconstruct S0, DoLP, and AoP," Opt. Express 27, 8566-8577 (2019)](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-27-6-8566)
