_base_ = [
    '../_base_/datasets/gleason_2019.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(type='VisionTransformer', config_name='R50-ViT-B_16')

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mDice')
