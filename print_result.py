import sys
import yaml
train_time = None
with open(sys.argv[1], 'r') as hf, open('eval_summary.yml', 'r') as ef:
    hdic = yaml.load(hf)
    edic = yaml.load(ef)
    print('train_IoU','valid_IoU','test_IoU',
          'train_mean_IoU','valid_mean_IoU','test_mean_IoU',sep='\t\t')
    print(max(hdic['acc']), max(hdic['val_acc']), hdic['test_acc'], 
          edic['train_mean_iou'], edic['valid_mean_iou'], edic['test_mean_iou'],
          sep='\t',end='\t')
    print(hdic['train_time'].split(':')[1])
