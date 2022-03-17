#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import glob
import sys
import time
from sklearn import metrics
import numpy
import pdb
from operator import itemgetter

# 卡阈值并返回EER及相应的threshold FAR FRR
def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    # 通过sklearn.metric.roc_curve计算ROC 得到阈值枚举列表及对应的FP(fall-out) TP(recall)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # 混淆矩阵
    #         预测 预测
    #          1   0
    #         -------
    # 标签 1 | TP  FN  P=TP+FN  -> TPR(recall) FN     FRR=FNR
    # 标签 0 | FP  TN  N=FP+TN  -> FPR         TNR    FAR=FPR
    #          P'  N'
             
    # TP: True Positive 真阳性 即 标签阳性(1) 且 预测阳性(1)
    # TPR(recall, sensitivity): True Postive Rate 真阳性率 TPR=TP/P=TP/(TP+FN) 即阳性样本中阳性预测的检出率 
    #
    # FP(false-alarm, I类错误): False Positive 假阳性 即 标签阴性(0) 但 预测阳性(1)
    # FPR(fall-out): False Positive Rate 假阳性率 FPR=FP/N=FP/(FP+TN) 即阴性样本中的错检率
    #
    # TN: True Negative 真阴性 即标签阴性(0) 且预测阴性(0)
    # TNR(specificity): True Negative Rate 真阴性率 TNR=TN/N=TN/(FP+TN) 即阴性样本中阴性预测的检出率 
    #
    # FN(miss, II类错误): False Negative 假阴性 即 标签阳性(1) 但 预测阴性(0)
    # FNR: False Negative Rate 假阴性率 FNR=FN/P=FN/(TP+FN) 即阳性样本中的错检率
     
    # P'=TP+FP 预测为阳性的总样本数
    # N'=FN+TN 预测为阴性的总样本数
    # 
    # FA: False Accept 错误接收 等价于FP
    # FAR: False Accept Rate 错误接收率 FAR=FPR
    # FR: False Reject 错误拒绝 等价于FN
    # FRR: False Reject Rate 错误拒绝率 FRR=FNR
    
    # Recall: 回召 Recall=TPR
    # Precison: 命中率 Precision=TP/P'=TP/(TP+FP) 即 阳性预测中的真阳性比例
    # F-score: Recall和Precision的加权调和平均 F-score=\frac{(a^2+1)Pe*R}{a^2(Pe+R)}
    # F1-score: F-score取a=1 F1-score=Pe*R/(Pe+R) <= 1
    
    # ROC曲线: Receiver Operation Curve 左上角(1,0)最优
    #   TPR  |   ---
    #        | /
    #        |/_____ FPR
    # DET曲线: Detection Error Trade-off 左下角(0,0)最优
    #   FRR(FNR=1-TPR)  ||   
    #                   | \
    #                   |___\__ FAR(FPR)
    # PR曲线: Precision-Recall 右上角(1,1)最优
    #   Precision  |---
    #              |    \
    #              |_____\ Recall
    # F1-score曲线: 顶部y=1最优
    #   F1 |   ---
    #      | /     \
    #      |/_______\ threshold
    
    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    # 找FRR=FAR的等错点
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])*100
    
    return (tunedThreshold, eer, fpr, fnr);

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
# 返回阈值枚举列表 及相应的FRR FAR列表
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold