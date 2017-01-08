# kaggle_facial_keypoints_detection
初学深度学习，自己在网上看到一个kaggle的博客，在此基础进行了摸索，最终进行实现和完善。

kaggle地址：https://www.kaggle.com/c/facial-keypoints-detection/details/getting-started-with-r


若kaggle的数据无法获取，可以去我的网盘中下载训练数据，然后修改成你自己的路径即可。地址：http://pan.baidu.com/s/1c2f1XSk


数据介绍：
training.csv: list of training 7049 images. Each row contains the (x,y) coordinates for 15 keypoints, and image data as row-ordered list of pixels.

test.csv: list of 1783 test images. Each row contains ImageId and image data as row-ordered list of pixels

submissionFileFormat.csv: list of 27124 keypoints to predict. Each row contains a RowId, ImageId, FeatureName, Location. FeatureName are "left_eye_center_x," "right_eyebrow_outer_end_y," etc. Location is what you need to predict. 

其中face文件夹中，是我将网站提供的数值，还原成了图片，保存在此文件夹.(注：数据集太大，无法传到git上，稍后将放到网盘上)
如果你们要在本地运行，请注意将代码中的路径都换成你们自己的路径，也就是数据的路径。

各python文件的含义：

face.py:关键点识别的网络模型

read_data.py:读取数据的文件，其中里面我还实现了一个将数据转成图片并存储的函数

train.py:读取训练集数据进行训练，将训练好的模型保存起来

test.py:读取保存的模型，并恢复网络，进行测试集的预测

predict.py:读取一张图片，通过模型，预测出15个关键点，并在图片上标识出来



参考网站：

http://blog.csdn.net/thriving_fcl/article/details/50909109

http://stackoverflow.com/questions/16720682/pil-cannot-write-mode-f-to-jpeg

http://www.bubuko.com/infodetail-1172011.html

http://www.cnblogs.com/meitian/p/3699223.html

http://www.cnblogs.com/yinxiangnan-charles/p/5928689.html

http://www.aichengxu.com/view/39904

http://blog.csdn.net/u012235274/article/details/52588690


总结：
 新手遇到了很多坑，包括中间还出了环境问题，如：
 PIL的IOError: decoder jpeg not available错误的排除方法

 如有问题，可以一起学习，讨论，qq:754379117
