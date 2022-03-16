

vox1_resl-sap_softmax:
	rm -rf /home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_softmax
	python3 ./trainSpeakerNet.py --gpu "2" --model ResNetSE34L --log_input True --encoder_type SAP --trainfunc softmax --save_path "/home/zzf/experiment-voxcelebtrainer/vox1_resl-sap_softmax" --nClasses 1211 --batch_size 200 --scale 30 --train_list "/home/zzf/dataset/vox1/train_list.txt" --test_list "/home/zzf/dataset/vox1/veri_test.txt" --train_path "/home/zzf/dataset/vox1/wav" --test_path "/home/zzf/dataset/vox1/wav_test"  