This repo use for detecting deepfake solution. 
Tech using is CNN + Multi-head Self-Attention. 
3 dataset used are: 140k face in Kaggle, CelebDF_v2 in Kaggle and CCDDF could be 
created by 5000 real images from 140k Face and 5000 fake images extract from video.

Activate Function is Focal Loss.
Cross Evaluate to Evaluate 3 Model after train, compare with each others.
Grad-CAM to extract where model see in the most and where was not be seen. 
