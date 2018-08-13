import yaml

with open('eval_summary.yml','r') as f:
    dic = yaml.load(f)
    valid_iou_arr = dic['valid_iou_arr']
    print(sorted( zip(valid_iou_arr,range(len(valid_iou_arr)) )))
    print('-------------------')
    train_iou_arr = dic['train_iou_arr']
    print(sorted( zip(train_iou_arr,range(len(train_iou_arr)) )))
