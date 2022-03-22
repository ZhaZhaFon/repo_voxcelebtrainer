
### before 0319

# Softmax
#vox1_resl-sap_softmax:
#	python3 ./trainSpeakerNet.py --gpu "2" --model ResNetSE34L --log_input True --encoder_type SAP --trainfunc softmax --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_softmax" --nClasses 1211 --batch_size 200 --train_list "/home/zzf/dataset/vox1/train_list.txt" --test_list "/home/zzf/dataset/vox1/veri_test.txt" --train_path "/home/zzf/dataset/vox1/wav" --test_path "/home/zzf/dataset/vox1/wav_test"  
#vox1_resl-sap_softmax_eval:
#	python3 ./trainSpeakerNet.py --gpu "2" --eval --model ResNetSE34L --log_input True --encoder_type SAP --trainfunc softmax --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_softmax" --nClasses 1211 --batch_size 200 --train_list "/home/zzf/dataset/vox1/train_list.txt" --test_list "/home/zzf/dataset/vox1/veri_test.txt" --train_path "/home/zzf/dataset/vox1/wav" --test_path "/home/zzf/dataset/vox1/wav_test"  

# AM-Softmax
#vox1_resl-sap_amsoftmax-m0.1:
#	python3 ./trainSpeakerNet.py --gpu "2" --trainfunc amsoftmax --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_amsoftmax-m0.1" --config ./base0317.yml
#vox1_resl-sap_amsoftmax-m0.1_eval:
#	python3 ./trainSpeakerNet.py --gpu "2" --eval --trainfunc amsoftmax --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_amsoftmax-m0.1" --config ./base0317.yml

# AAM-Softmax
#vox1_resl-sap_aamsoftmax-m0.1s30:
#	python3 trainSpeakerNet.py --gpu "3" --trainfunc aamsoftmax --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_aamsoftmax-m0.1s30" --config ./base0317.yml
#vox1_resl-sap_aamsoftmax-m0.1s30_eval:
#	python3 trainSpeakerNet.py --gpu "3" --eval --trainfunc aamsoftmax --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_aamsoftmax-m0.1s30" --config ./base0317.yml

### after 0319

# Softmax
vox1_resl-sap_softmax:
	python3 trainSpeakerNet.py --gpu "2" --trainfunc softmax --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_softmax" --config "./base0319.yml"
vox1_resl-sap_softmax-eval:
	python3 trainSpeakerNet.py --gpu "2" --eval --trainfunc softmax --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_softmax" --config "./base0319.yml"

# AM-Softmax
vox1_resl-sap_amsoftmax-m0.1:
	python3 trainSpeakerNet.py --gpu "3" --trainfunc amsoftmax --margin 0.1 --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_amsoftmax-m0.1" --config "./base0319.yml"
vox1_resl-sap_amsoftmax-m0.1_eval:
	python3 trainSpeakerNet.py --gpu "3" --eval --trainfunc amsoftmax --margin 0.1 --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_amsoftmax-m0.1" --config "./base0319.yml"

# AAM-Softmax
vox1_resl-sap_aamsoftmax-m0.1s30:
	python3 trainSpeakerNet.py --gpu "3" --trainfunc aamsoftmax --margin 0.1 --scale 30 --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_aamsoftmax-m0.1s30" --config ./base0319.yml
vox1_resl-sap_aamsoftmax-m0.1s30_eval:
	python3 trainSpeakerNet.py --gpu "3" --eval --trainfunc aamsoftmax --margin 0.1 --scale 30 --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_aamsoftmax-m0.1s30" --config ./base0319.yml

# triplet
# rank-based hard negative mining
vox1_resl-sap_triplet-m0.1-rank10prob0.5:
	python3 trainSpeakerNet.py --gpu "2" --trainfunc triplet --margin 0.1 --nPerSpeaker 2 --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_triplet-m0.1-rank10prob0.5" --config "./base0319.yml"
vox1_resl-sap_triplet-m0.1-rank10prob0.5_eval:
	python3 trainSpeakerNet.py --gpu "2" --eval  --trainfunc triplet --margin 0.1 --nPerSpeaker 2 --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_triplet-m0.1-rank10prob0.5" --config "./base0319.yml"
# semi-hard hard negative mining
vox1_resl-sap_triplet-m0.1-semi10:
	python3 trainSpeakerNet.py --gpu "3" --trainfunc triplet --margin 0.1 --nPerSpeaker 2 --hard_rank -1 --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_triplet-m0.1-semi10" --config "./base0319.yml"
vox1_resl-sap_triplet-m0.1-semi10_eval:
	python3 trainSpeakerNet.py --gpu "3" --eval --trainfunc triplet --margin 0.1 --nPerSpeaker 2 --hard_rank -1 --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_triplet-m0.1-semi10" --config "./base0319.yml"

# prototypical
vox1_resl-sap_proto-s5q1:
	python3 trainSpeakerNet.py --gpu "2" --trainfunc proto --nPerSpeaker 6 --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_proto-s5q1" --config "./base0319.yml"
vox1_resl-sap_proto-s5q1_eval:
	python3 trainSpeakerNet.py --gpu "2" --eval --trainfunc proto --nPerSpeaker 6 --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_proto-s5q1" --config "./base0319.yml"

# angular prototypical
# prototypical
vox1_resl-sap_angleproto-s5q1:
	python3 trainSpeakerNet.py --gpu "2" --trainfunc angleproto --nPerSpeaker 6 --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_angleproto-s5q1" --config "./base0319.yml"
vox1_resl-sap_angleproto-s5q1_eval:
	python3 trainSpeakerNet.py --gpu "2" --eval --trainfunc angleproto --nPerSpeaker 6 --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_angleproto-s5q1" --config "./base0319.yml"

# GE2E
vox1_resl-sap_ge2e-m5:
	python3 trainSpeakerNet.py --gpu "3" --trainfunc ge2e --nPerSpeaker 5 --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_ge2e-m5" --config "./base0319.yml"
vox1_resl-sap_ge2e-m5_eval:
	python3 trainSpeakerNet.py --gpu "3" --eval --trainfunc ge2e --nPerSpeaker 5 --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_ge2e-m5" --config "./base0319.yml"