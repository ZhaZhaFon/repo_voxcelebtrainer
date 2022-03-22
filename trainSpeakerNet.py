#!/usr/bin/python
#-*- coding: utf-8 -*-

# original codebase: https://github.com/clovaai/voxceleb_trainer
# edited and re-distributed by Zifeng Zhao @ Peking University
# 2022.03

import sys, time, os, argparse, socket
import yaml
import numpy
import pdb
import torch
import glob
import zipfile
import warnings
import datetime
import SpeakerNet
import tuneThreshold
import DatasetLoader
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments 运行参数解析
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "SpeakerNet");

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file'); # yaml配置文件路径

## Data loader 数据处理相关
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training');
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing; 0 uses the whole files');
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch'); # batch_size
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads');
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator');

## Training details 训练相关
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs'); # 测试/保存模型周期
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs'); # epoch
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function'); # loss: softmax amsoftmax aamsoftmax triplet proto softmaxproto angleproto ge2e

## Optimizer 梯度下降
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam'); # optimizer
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler'); # scheduler
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate'); # learning_rate
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');

## Loss functions 损失函数
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions'); # rank-based negative mining概率
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions'); # 若取>=0则采用rank-based negative mining; 若取<0则采用semi-hard negative mining 
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions'); # for AM-Softmax AAM-Softmax Triplet
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions'); # for AM-Softmax AAM-Softmax
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses'); # batch_size = nPerSpeaker x num_spk_in_batch # Triplet
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses'); # softmax分类头

## Evaluation parameters DCF参数
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker');
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection');
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection');

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights');
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs'); # 实验保存路径

## Training and test data 数据路径
parser.add_argument('--train_list',     type=str,   default="data/train_list.txt",  help='Train list'); # 训练集的train_list.txt列表
parser.add_argument('--test_list',      type=str,   default="data/test_list.txt",   help='Evaluation list'); # 测试集的test_list.txt列表
parser.add_argument('--train_path',     type=str,   default="data/voxceleb2", help='Absolute path to the train set'); # 训练集绝对路径
parser.add_argument('--test_path',      type=str,   default="data/voxceleb1", help='Absolute path to the test set'); # 测试集绝对路径
parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set');
parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set');

## Model definition 模型参数
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks');
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition'); # model: ResNetSE34L ResNetSE34V2 VGGVox
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder');
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer'); # SpeakerEmbedding维度

## For test only 测试评估
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only') # eval模式

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text');
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

# 新增参数
parser.add_argument("--gpu",            type=str,   default="0",    required=True, help="available GPU")
parser.add_argument("--tensorboard",    type=bool,  default=False,  help="write logs with tensorboard")

print('')
print('# 解析parser运行参数')
args = parser.parse_args();

## Parse YAML 可在config参数项中传入yaml配置文件(或者接使用args运行参数)
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError
if args.config is not None:
    print('# 解析yaml配置文件')
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

## Try to import NSML
# Naver Corp的内部工具 可以不用
# https://n-clair.github.io/ai-docs/_build/html/en_US/index.html
try:
    import nsml
    from nsml import HAS_DATASET, DATASET_PATH, PARALLEL_WORLD, PARALLEL_PORTS, MY_RANK
    from nsml import NSML_NFS_OUTPUT, SESSION_NAME
except:
    pass;

warnings.simplefilter("ignore")

## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script 训练主进程
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):
    
    print('# 进入主进程...')
    
    args.gpu = gpu

    ## Load models
    print('   # 加载网络结构')
    model = SpeakerNet.SpeakerNet(**vars(args));

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        torch.cuda.set_device(args.gpu)
        model = SpeakerNet.WrappedModel(model).cuda(args.gpu)

    it = 1
    eers = [100];

    #if args.gpu == 0:
    if True:
        ## Write args to scorefile
        scorefile   = open(args.result_save_path+"/scores.txt", "a+");

    ## Initialise trainer and data loader
    print('   # 加载数据...')
    print('      # 加载train_dataset')
    train_dataset = DatasetLoader.train_dataset_loader(**vars(args))
    print('      # 加载train_sampler')
    train_sampler = DatasetLoader.train_dataset_sampler(train_dataset, **vars(args))
    print('      # 加载train_loader')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=DatasetLoader.worker_init_fn,
        drop_last=True,
    )

    print('   # 加载SpeakerNet.ModelTrainer')
    # trainLoader = get_data_loader(args.train_list, **vars(args));
    trainer     = SpeakerNet.ModelTrainer(model, **vars(args))

    ## Load model weights
    print('   # 检索断点')
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if(args.initial_model != ""):
        print("      # 从给定的{}加载模型参数".format(args.initial_model));
        trainer.loadParameters(args.initial_model);
        #print("Model {} loaded!".format(args.initial_model));
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1]);
        print("      # 从{}加载断点".format(modelfiles[-1]));
        #print("Model {} loaded from previous state!".format(modelfiles[-1]));
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
        print("      # 从epoch{}继续".format(it))
    else:
        print("      # 从0起训");

    for ii in range(1,it):
        trainer.__scheduler__.step()
    
    ## Evaluation code - must run on single GPU 评估模式
    if args.eval == True:
        
        print('')
        print('### START EVALUATION ###')

        pytorch_total_params = sum(p.numel() for p in model.module.__S__.parameters())

        print('Total parameters: ',pytorch_total_params)
        print('Test list',args.test_list)
        
        # 返回测试列表上所有测试对的打分结果sc和对应标签lab
        print('# SpeakerEmbedding推理&打分...')
        sc, lab, _ = trainer.evaluateFromList(**vars(args))

        #if args.gpu == 0:
        if True:
            
            print('\n# 卡阈值...', end='')

            # 卡阈值 返回[thr, EER, FAR, FRR]
            result = tuneThreshold.tuneThresholdfromScore(sc, lab, [1, 0.1]);

            fnrs, fprs, thresholds = tuneThreshold.ComputeErrorRates(sc, lab)
            mindcf, threshold = tuneThreshold.ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EvalEER {:2.4f}%".format(result[1]), "MinDCF {:2.5f}".format(mindcf));

            #if ("nsml" in sys.modules) and args.gpu == 0:
            if ("nsml" in sys.modules):
                training_report = {};
                training_report["summary"] = True;
                training_report["epoch"] = it;
                training_report["step"] = it;
                training_report["val_eer"] = result[1];
                training_report["val_dcf"] = mindcf;

                nsml.report(**training_report);

        print('### EVALUATION COMPLETED ###')
        print('')
        return
    
    # 训练模式

    ## Save training code and params 备份训练程序和训练参数
    #if args.gpu == 0:
    if True:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(args.result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(args.result_save_path + '/run%s.cmd'%strtime, 'w') as f:
            f.write('%s'%args)
            
    # tensorboard
    if args.tensorboard:
        tb_writer = SummaryWriter(os.path.join(args.result_save_path, 'tb_logs'))

    ## Core training script
    print('')
    print('### START TRAINING ###')
    for it in range(it,args.max_epoch+1):

        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        print(f'\n# epoch {it} - train...')
        #loss, traineer = trainer.train_network(train_loader, verbose=(args.gpu == 0));
        loss, traineer = trainer.train_network(train_loader, verbose=True);

        # if args.gpu == 0:
        if True:
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, TrainEER/TrainAcc {:2.2f}, TrainLOSS {:f}, LR {:f}".format(it, traineer, loss, max(clr)));
            scorefile.write("Epoch {:d}, TrainEER/TrainAcc {:2.2f}, TrainLOSS {:f}, LR {:f} \n".format(it, traineer, loss, max(clr)));
            
            if args.tensorboard:
                tb_writer.add_scalar('train/loss', loss, it)
                tb_writer.add_scalar('train/EER_or_Acc', traineer, it)
                tb_writer.add_scalar('train/learning_rate', max(clr), it)

        if it % args.test_interval == 0:
            
            print(f'\n# epoch {it} - test_interval...')
            # 返回测试列表上所有测试对的打分结果sc和对应标签lab
            sc, lab, _ = trainer.evaluateFromList(**vars(args))

            #if args.gpu == 0:
            if True:
                
                # 卡阈值 返回[thr, EER, FAR, FRR]
                result = tuneThreshold.tuneThresholdfromScore(sc, lab, [1, 0.1]);

                fnrs, fprs, thresholds = tuneThreshold.ComputeErrorRates(sc, lab)
                mindcf, threshold = tuneThreshold.ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

                eers.append(result[1])

                print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, EvalEER {:2.4f}%, MinDCF {:2.5f}".format(it, result[1], mindcf));
                scorefile.write("Epoch {:d}, EvalEER {:2.4f}%, MinDCF {:2.5f}\n".format(it, result[1], mindcf));

                trainer.saveParameters(args.model_save_path+"/model%09d.model"%it);

                with open(args.model_save_path+"/model%09d.eer"%it, 'w') as eerfile:
                    eerfile.write('{:2.4f}'.format(result[1]))

                scorefile.flush()
                
                if args.tensorboard:
                    tb_writer.add_scalar('val/EER', result[1], it)
                    tb_writer.add_scalar('val/MinDCF', mindcf, it)

        #if ("nsml" in sys.modules) and args.gpu == 0:
        if ("nsml" in sys.modules):
            training_report = {};
            training_report["summary"] = True;
            training_report["epoch"] = it;
            training_report["step"] = it;
            training_report["train_loss"] = loss;
            training_report["min_eer"] = min(eers);

            nsml.report(**training_report);

    #if args.gpu == 0:
    if True:
        scorefile.close();
        
    print('### TRAINING COMPLETED ###')
    print('')


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function 主函数入口
## ===== ===== ===== ===== ===== ===== ===== =====


def main():

    if ("nsml" in sys.modules) and not args.eval:
        args.save_path  = os.path.join(args.save_path,SESSION_NAME.replace('/','_'))

    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args)) # 多卡
    else:
        print('# 使用单GPU: {}'.format(args.gpu))
        main_worker(int(args.gpu), None, args) # 单卡
        #main_worker(0, None, args)


if __name__ == '__main__':
    main()
    
    from IPython import embed
    embed()
    print('done.')