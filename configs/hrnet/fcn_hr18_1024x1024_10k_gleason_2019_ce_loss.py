_base_ = './fcn_hr18_1024x1024_20k_gleason_2019_ce_loss.py'
runner = dict(type='IterBasedRunner', max_iters=10000)
evaluation = dict(
    interval=1000, metric='mDice', save_best='mDice', rule='greater')
