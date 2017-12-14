# jdd_pig_recognition_contest
The python source code of JDD (Jing Dong Discovery) pig recognition contest (https://jddjr.jd.com/item/4).

### 1. 软件环境
* tensorflow：0.10.0rc0
* keras：2.1.2
* opencv：3.3.1
* pandas：0.19.2

### 2. 执行流程
* video2image.py：实现猪mp4 videos到images的转化，以及将images分为train set和valid set。
* jdd_pig_recognition.py：首先，通过vgg16生成train set、valid set和test set的1000维特征；然后构造dnn做多分类（30类）；最后输出test set的预测结果。

### 3. 经验总结
* 重要参数：样本量（包括train和valid样本的比例，若样本量够大，10：1或train更多）、epoch+batch（epoch一定要做到使训练指标稳定下来，batch的可尝试4、16、64、128、256等）、learning rate、optimizer等。
