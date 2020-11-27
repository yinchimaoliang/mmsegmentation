_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/pathology_full.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

model = dict(
    decode_head=dict(num_classes=5), auxiliary_head=dict(num_classes=5))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)

evaluation = dict(interval=500, metric='mDice')
