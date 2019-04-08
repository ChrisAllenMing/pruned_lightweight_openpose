from threading import Timer
from datetime import datetime
import os
import time
import val_prune

epochId = 33
checkpoints_folder='lr4e-5_thre0.3_1_checkpoints'
LOG_FOUT = open(os.path.join(checkpoints_folder, 'log_val.txt'), 'w')
while(1):
    checkpoint_path = '{}/checkpoint_epoch_{}.pth.tar'.format(checkpoints_folder, epochId)
    if(os.path.exists(checkpoint_path)):
        results = val_prune.main('./coco/annotations/person_keypoints_val2017.json', './coco/val2017',checkpoint_path)
        results.summarize()
        epochId+=1
    


