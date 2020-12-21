import argparse
import glob
import os.path as osp
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--work-dir', help='path of log')
    args = parser.parse_args()

    return args


def _parse_log(work_dir):
    log_paths = glob.glob(osp.join(work_dir, '*.log'))
    log_paths.sort()
    log_path = log_paths[-1]
    losses = dict()
    loss_iters = list()
    APs = dict()
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'loss:' in line:
                for item in line.split(', '):
                    if 'Iter' in item:
                        loss_iters.append(int(item[item.find('[')+1: item.find('/')]))
                    if 'loss' in item:
                        name, value = item.split(': ')
                        value = float(value.split('\n')[0])
                        if name not in losses.keys():
                            losses[name] = [value]
                        else:
                            losses[name].append(value)
    return loss_iters, losses

def _draw_log_data(iters, data, name, work_dir):
    for name in data.keys():
        plt.plot(iters, data[name], label=name)
    plt.xlabel('iters')
    plt.ylabel(name)
    plt.legend()
    plt.savefig(osp.join(work_dir, f'{name}.png'))


def main():
    args = parse_args()
    work_dir = args.work_dir
    loss_iters, losses = _parse_log(work_dir)
    _draw_log_data(loss_iters, losses, 'loss', work_dir)


if __name__ == '__main__':
    main()