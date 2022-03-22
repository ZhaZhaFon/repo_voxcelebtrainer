
# repo_voxcelebtrainer

This repository contains the framework for training speaker recognition models described in the paper '_In defence of metric learning for speaker recognition_'.

本项目代码出自论文[1], 用于实验不同的深度度量学习方法训练xvector用于说话人识别的效果. 在[原项目](https://github.com/clovaai/voxceleb_trainer)的基础上, 做了一些配置方便使用, 并加入了一些注释.

### Dependencies 环境配置

运行以下命令一键配置环境
```
pip install -r requirements.txt
```

### 文件组织

```
voxceleb_trainer/ 
    models/ 各网络结构
      ResNetSE34L.py
    loss/ 各损失函数
      softmax.py 
      amsoftmax.py
      aamsoftmax.py
      triplet.py
    DatasetLoader.py 数据IO
    SpeakerNet.py SpeakerNet类
    trainSpeakerNet.py 训练主程序
    tuneThreshold.py 卡阈值及指标计算相关
```

### 主要实现逻辑

* SpeakerNet.ModelTrainer单epoch训练逻辑  

trainSpeakerNet.py主程序中对象化class SpeakerNet.ModelTrainer, 并通过其train_network()方法进行单epoch训练. 对每批数据, 通过class SpeakerNet.SpeakerNet模型的forward()方法进行前向传播
```
def train_network(self, loader, verbose):

        self.__model__.train();

        stepsize = loader.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0;    # EER or accuracy

        tstart = time.time()
        
        for data, data_label in loader:

            data    = data.transpose(1,0)

            self.__model__.zero_grad();

            label   = torch.LongTensor(data_label).cuda()

            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__(data, label)
                self.scaler.scale(nloss).backward();
                self.scaler.step(self.__optimizer__);
                self.scaler.update();       
            else:
                # 前向传播
                nloss, prec1 = self.__model__(data, label)
                # 反向传播
                nloss.backward();
                self.__optimizer__.step();


            loss    += nloss.detach().cpu().item();
            top1    += prec1.detach().cpu().item();
            counter += 1;
            index   += stepsize;

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__()*loader.batch_size));
                sys.stdout.write("Loss {:f} TrainEER/TrainAcc {:2.3f}% - {:.2f} Hz ".format(loss/counter, top1/counter, stepsize/telapsed));
                sys.stdout.flush();

            if self.lr_step == 'iteration': self.__scheduler__.step()

        if self.lr_step == 'epoch': self.__scheduler__.step()
        
        return (loss/counter, top1/counter);
```

* class SpeakerNet.SpeakerNet模型  

SpeakerNet.SpeakerNet模型包括SpeakerEncoder子模型和LossFunction子模型, 分别实现Embedding推理和Loss计算
```
class SpeakerNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__();

        # SpeakerEncoder模型
        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs);

        # Loss函数
        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs);

        self.nPerSpeaker = nPerSpeaker

    def forward(self, data, label=None):

        # SpeakerEncoder子模型前向传播 得到SpeakerEmbedding
        data    = data.reshape(-1,data.size()[-1]).cuda() 
        outp    = self.__S__.forward(data)

        if label == None:
            return outp

        else:

            outp    = outp.reshape(self.nPerSpeaker,-1,outp.size()[-1]).transpose(1,0).squeeze(1)
            
            # SpeakerEmbedding送给Loss子模型计算loss和precision
            nloss, prec1 = self.__L__.forward(outp,label)

            return nloss, prec1
```

* SpeakerEncoder子模型  
网络以类形式定义. forward()方法中实现输入到SpeakerEmbedding的前向传播. 略

* Loss子模型  
以softmax.py为例: Loss以类似网络的方式定义一个类class LossFunction来实现, 以forward()作为接口, 接收SpeakerEncoder网络的输出(Speaker Embedding), 经过全连接FC规整后输出到nOut个分类头, 多分类预测结果和多分类标签送入多分类交叉熵torch.nn.CrossEntropyLoss计算分类损失(实现上与先进行LogSoftmax规整再送负对数似然NLLLoss计算多分类损失等价). 最后loss和precision返回给SpeakerEncoder子模型.
```
class LossFunction(nn.Module):
	def __init__(self, nOut, nClasses, **kwargs):
	    super(LossFunction, self).__init__()

	    self.test_normalize = True
	    
	    self.criterion  = torch.nn.CrossEntropyLoss()
	    self.fc 		= nn.Linear(nOut,nClasses)

	    print('Initialised Softmax Loss')

	def forward(self, x, label=None):

		x 		= self.fc(x)
		nloss   = self.criterion(x, label)
		prec1	= accuracy(x.detach(), label.detach(), topk=(1,))[0]

		return nloss, prec1
```

### 改动

* 加入了Makefile方便运行
* 可以指定GPU进行单卡训练
* 设定了几个基本配置的yaml
* 修正了triplet.py
* 加入了tensorboard打log
* 修正了proto.py

### TODO LIST
* angleproto

### Data preparation

The following script can be used to download and prepare the VoxCeleb dataset for training.

```
python ./dataprep.py --save_path data --download --user USERNAME --password PASSWORD 
python ./dataprep.py --save_path data --extract
python ./dataprep.py --save_path data --convert
```
In order to use data augmentation, also run:

```
python ./dataprep.py --save_path data --augment
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.

### Training examples

- AM-Softmax:
```
python ./trainSpeakerNet.py --model ResNetSE34L --log_input True --encoder_type SAP --trainfunc amsoftmax --save_path exps/exp1 --nClasses 5994 --batch_size 200 --scale 30 --margin 0.3
```

- Angular prototypical:
```
python ./trainSpeakerNet.py --model ResNetSE34L --log_input True --encoder_type SAP --trainfunc angleproto --save_path exps/exp2 --nPerSpeaker 2 --batch_size 200
```

The arguments can also be passed as `--config path_to_config.yaml`. Note that the configuration file overrides the arguments passed via command line.

### Pretrained models

A pretrained model, described in [1], can be downloaded from [here](http://www.robots.ox.ac.uk/~joon/data/baseline_lite_ap.model).

You can check that the following script returns: `EER 2.1792`. You will be given an option to save the scores.

```
python ./trainSpeakerNet.py --eval --model ResNetSE34L --log_input True --trainfunc angleproto --save_path exps/test --eval_frames 400 --initial_model baseline_lite_ap.model
```

A larger model trained with online data augmentation, described in [2], can be downloaded from [here](http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model). 

The following script should return: `EER 1.1771`.

```
python ./trainSpeakerNet.py --eval --model ResNetSE34V2 --log_input True --encoder_type ASP --n_mels 64 --trainfunc softmaxproto --save_path exps/test --eval_frames 400  --initial_model baseline_v2_ap.model
```

### Implemented loss functions
```
Softmax (softmax)
AM-Softmax (amsoftmax)
AAM-Softmax (aamsoftmax)
GE2E (ge2e)
Prototypical (proto)
Triplet (triplet)
Angular Prototypical (angleproto)
```

### Implemented models and encoders
```
ResNetSE34L (SAP, ASP)
ResNetSE34V2 (SAP, ASP)
VGGVox40 (SAP, TAP, MAX)
```

### Data augmentation

`--augment True` enables online data augmentation, described in [2].

### Adding new models and loss functions

You can add new models and loss functions to `models` and `loss` directories respectively. See the existing definitions for examples.

### Accelerating training

- Use `--mixedprec` flag to enable mixed precision training. This is recommended for Tesla V100, GeForce RTX 20 series or later models.

- Use `--distributed` flag to enable distributed training.

  - GPU indices should be set before training using the command `export CUDA_VISIBLE_DEVICES=0,1,2,3`.

  - If you are running more than one distributed training session, you need to change the `--port` argument.

### Data

The [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) datasets are used for these experiments.

The train list should contain the identity and the file path, one line per utterance, as follows:
```
id00000 id00000/youtube_key/12345.wav
id00012 id00012/21Uxsk56VDQ/00001.wav
```

The train list for VoxCeleb2 can be download from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt) and the
test list for VoxCeleb1 from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt).

### Replicating the results from the paper

1. Model definitions
  - `VGG-M-40` in [1] is `VGGVox` in the repository.
  - `Thin ResNet-34` in [1] is `ResNetSE34` in the repository.
  - `Fast ResNet-34` in [1] is `ResNetSE34L` in the repository.
  - `H / ASP` in [2] is `ResNetSE34V2` in the repository.

2. For metric learning objectives, the batch size in the paper is `nPerSpeaker` multiplied by `batch_size` in the code. For the batch size of 800 in the paper, use `--nPerSpeaker 2 --batch_size 400`, `--nPerSpeaker 3 --batch_size 266`, etc.

3. The models have been trained with `--max_frames 200` and evaluated with `--max_frames 400`.

4. You can get a good balance between speed and performance using the configuration below.

```
python ./trainSpeakerNet.py --model ResNetSE34L --trainfunc angleproto --batch_size 400 --nPerSpeaker 2 
```

### Citation

Please cite [1] if you make use of the code. Please see [here](References.md) for the full list of methods used in this trainer.

[1] _In defence of metric learning for speaker recognition_
```
@inproceedings{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  booktitle={Interspeech},
  year={2020}
}
```

[2] _Clova baseline system for the VoxCeleb Speaker Recognition Challenge 2020_
```
@article{heo2020clova,
  title={Clova baseline system for the {VoxCeleb} Speaker Recognition Challenge 2020},
  author={Heo, Hee Soo and Lee, Bong-Jin and Huh, Jaesung and Chung, Joon Son},
  journal={arXiv preprint arXiv:2009.14153},
  year={2020}
}
```

### License
```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
