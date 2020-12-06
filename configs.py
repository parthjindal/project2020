adam = {
    'lr':1e-5,
    'betas':(0.9,0.999),
    'eps':1e-8 ,
    'weight_decay':0
}

sgd = {
    'lr':1e-4,
    'momentum':0,
    'weight_decay':0
}

config = {
    'optimizer':'adam',
    'mode':'train',
    'adam':adam,
    'sgd':sgd,
    'episodes':1000,
    'batchsize':256,
    'tensorboard_dir':'./tb/',
    'logdir':'',
    'model_dir':'./models/',
    'GPUS':0,
    'cpus':4,
    'device':'cuda:0',
    'filter_length':8,
    'padding_value':0,
}

