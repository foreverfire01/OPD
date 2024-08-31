data_root = '/home/cv/Data/ASDD/all_104070/' 
pretrained = '/home/cv/checkpoints/download/swin_base_patch4_window12_384_22k.pth'
load_from = None
resume_from = None

# data_root = '/home/cv/Data/DOTA/DOTA_ship/'        
# pretrained = '/home/cv/checkpoints/download/swin_base_patch4_window12_384_22k.pth'
# load_from = '/home/cv/checkpoints/ASDD/retinanet_768_fpn4_all_opd_12e_ship_4321all_fromzero_newfl_fc2075/epoch_12.pth'
# resume_from = None

max_epochs = 12
num_last_epochs =0


img_scale = (768, 768)

train_batch_size_per_gpu = 4
train_num_workers = 4

checkpoint_interval = 1
log_interval = 100
eval_interval = 1

filter_empty_gt_train = True
filter_empty_gt_val = False
filter_empty_gt_test = False     # modify
# optimizer
optimizer = dict(
    type='SGD', 
    lr=0.001, 
    momentum=0.9, 
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])  
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)


#-------------------------------------------------------------------------------------------------


# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        # _delete_=True,
        type='OPDSwinTransformer',
        embed_dims=128,                        # modify
        depths=[2, 2, 18, 2],                  # modify
        num_heads=[4, 8, 16, 32],              # modify
        window_size=12,                        # modify
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),                 # modify
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],   # modify
        out_channels=256,
        start_level=1,                       # modify
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=3.0)),   # ???
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,              # VIP
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))



#---COCO
# dataset settings
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),    # add
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=img_scale,
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),   # modify
#         ])
# ]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),   
        ])
]

data = dict(
    samples_per_gpu=train_batch_size_per_gpu,
    workers_per_gpu=train_num_workers,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',                # modify
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline,
        filter_empty_gt=filter_empty_gt_train),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',                 # modify
        img_prefix=data_root + 'images/', 
        pipeline=test_pipeline,
        filter_empty_gt=filter_empty_gt_val),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',               # modify
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline,
        filter_empty_gt=filter_empty_gt_test))

evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=eval_interval,
    # dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox')


#--- runtime
checkpoint_config = dict(interval=checkpoint_interval)
# yapf:disable
log_config = dict(
    interval=log_interval,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'