#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from tuneThreshold import tuneThresholdfromScore
import random

class LossFunction(nn.Module):

    def __init__(self, hard_rank=0, hard_prob=0, margin=0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.hard_rank  = hard_rank
        self.hard_prob  = hard_prob
        self.margin     = margin

        print('Initialised Triplet Loss')

    def forward(self, x, label=None):
    
        assert x.size()[1] == 2
        
        out_anchor      = F.normalize(x[:,0,:], p=2, dim=1)
        out_positive    = F.normalize(x[:,1,:], p=2, dim=1) # 需要 --nPerSpeaker 2
        stepsize        = out_anchor.size()[0]

        #  out_anchor的batch_size(N)条样本与out_positive的batch_size(N)条样本进行碰撞得到(N,N)距离矩阵
        #  (N, D)                                   (N, D, 1)                    (1, D, N)
        #output      = -1 * (F.pairwise_distance(out_anchor.unsqueeze(-1),out_positive.unsqueeze(-1).transpose(0,2))**2)
        #  (N, N)                                  (N, 1, D)                    (1, N, D)
        output      = -1 * (F.pairwise_distance(out_anchor.unsqueeze(1),out_positive.unsqueeze(0))**2)

        negidx      = self.mineHardNegative(output.detach())

        out_negative = out_positive[negidx,:]

        labelnp     = numpy.array([1]*len(out_positive)+[0]*len(out_negative))

        ## calculate distances
        # anchor-postive pair的sqrt(L2) distance
        pos_dist    = F.pairwise_distance(out_anchor,out_positive)
        # anchor-negative pair的sqrt(L2) distance
        neg_dist    = F.pairwise_distance(out_anchor,out_negative)

        ## loss function
        nloss   = torch.mean(F.relu(torch.pow(pos_dist, 2) - torch.pow(neg_dist, 2) + self.margin))

        scores = -1 * torch.cat([pos_dist,neg_dist],dim=0).detach().cpu().numpy()

        # EER
        errors = tuneThresholdfromScore(scores, labelnp, []);

        #return nloss, errors[1]
        return nloss, torch.tensor(errors[1], device=nloss.device)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Hard negative mining 负样本挖掘
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def mineHardNegative(self, output):

        negidx = []

        #          (1, N)
        for idx, similarity in enumerate(output):

            # 对相似度进行升序 由于output是-1*L2 因此不相似在前 相似在后
            simval, simidx = torch.sort(similarity,descending=True) # L2取相反数后 similarity越大越接近

            if self.hard_rank < 0:

                ## Semi-hard negative mining
                # 负样本挖掘方案1 semi-hard[1]
                # [1] F. Schoff, et al. FaceNet: A Unified Embedding for Face Recognition and Clustering. IEEE Conf. on CVPR 2015
                
                # easy case (L2为相似度)
                # d(a,p) + margin < d(a,n)              Eq 1
                # semi-hard case (L2为相似度)
                # d(a,n) - margin < d(a,p) < d(a,n)     Eq 2
                #    OR
                # d(a,p) < d(a,n) < d(a,p) + margin     Eq 3
                # hard case (L2为相似度)
                # d(a,p) > d(a,n)                       Eq4

                # 筛选semi-hard样本
                # similarity[idx]: anchor-positive相似度d(a,p)
                # simval: anchor-negative相似度d(a,n)
                #                       d(a,p) - margin < d(a,n) < d(a,p)  由于output是-1*L2 故此处恰好与Eq3符号相反
                semihardidx = simidx[(similarity[idx] - self.margin < simval) &  (simval < similarity[idx])]

                if len(semihardidx) == 0:
                    # 没有挖掘到semi-hard样本 则从随机抽一条
                    negidx.append(random.choice(simidx))
                else:
                    # 挖掘到semi-hard样本 从semi-hard样本中随机抽一条
                    negidx.append(random.choice(semihardidx))

            else:

                ## Rank-based negative mining
                # 负样本挖掘方案2 rank-based
                
                # 负样本挖掘的标签不能与anchor相同 否则是正样本
                simidx = simidx[simidx!=idx]

                # 以--hard_prob为概率进行Rank-based negative mining
                if random.random() < self.hard_prob:
                    # Rank-based negative mining: 从前10hard中随机抽取
                    negidx.append(simidx[random.randint(0, self.hard_rank)])
                else:
                    # 随机抽取: 从(N-1)条中随机抽取
                    negidx.append(random.choice(simidx))

        return negidx