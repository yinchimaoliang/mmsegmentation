import torch

def get_one_hot(label, N):
    new_label = label.clone()
    # Remove label == 255
    new_label[new_label == 255] = 0
    size = list(new_label.size())
    new_label = new_label.view(-1)  # reshape 为向量
    ones = torch.eye(N).type_as(new_label)
    ones = ones.index_select(0, new_label)  # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size).permute([0, 3, 1, 2]).float()