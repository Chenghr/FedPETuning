import math
import numpy as np

"""
This code snippets come from https://github.com/FedML-AI/FedNLP/blob/master/data/advanced_partition/niid_label.py
"""

def dynamic_batch_fill(label_index_tracker, label_index_matrix,
                       remaining_length, current_label_id):
    """
    动态批量填充函数

    参数:
    ------------------------------------------------------------------------
    label_index_tracker : 1d numpy array，跟踪每个标签使用了多少数据
    label_index_matrix : 2d array，每个标签的索引列表
    remaining_length : int，当前分区客户端列表中剩余的空间
    current_label_id : int，当前轮的标签ID
    ------------------------------------------------------------------------

    返回:
    ---------------------------------------------------------
    label_index_offset: dict，字典的键是标签ID，值是与该键关联的偏移量
    ----------------------------------------------------------
    """
    remaining_unfiled = remaining_length
    label_index_offset = {}
    label_remain_length_dict = {}
    total_label_remain_length = 0

    # 计算所有剩余标签的总数以及每个标签的剩余长度
    for label_id, label_list in enumerate(label_index_matrix):
        if label_id == current_label_id:
            label_remain_length_dict[label_id] = 0
            continue
        label_remaining_count = len(label_list) - label_index_tracker[label_id]
        if label_remaining_count > 0:
            total_label_remain_length = (total_label_remain_length +
                                         label_remaining_count)
        else:
            label_remaining_count = 0
        label_remain_length_dict[label_id] = label_remaining_count
    length_pointer = remaining_unfiled

    if total_label_remain_length > 0:
        # 根据剩余长度对标签进行排序
        label_sorted_by_length = {
            k: v
            for k, v in sorted(label_remain_length_dict.items(),
                               key=lambda item: item[1])
        }
    else:
        label_index_offset = label_remain_length_dict
        return label_index_offset
    
    # 对每个标签计算偏移量，按照剩余标签的分布将其向前移动
    for label_id in label_sorted_by_length.keys():
        fill_count = math.ceil(label_remain_length_dict[label_id] /
                               total_label_remain_length * remaining_length)
        fill_count = min(fill_count, label_remain_length_dict[label_id])
        offset_forward = fill_count
        # 如果剩余的空间不足以容纳所有的偏移量，则将其设置为0
        if length_pointer - offset_forward <= 0 and length_pointer > 0:
            label_index_offset[label_id] = length_pointer
            length_pointer = 0
            break
        else:
            length_pointer -= offset_forward
            label_remain_length_dict[label_id] -= offset_forward
        label_index_offset[label_id] = offset_forward

    # 如果还有未填充的空间
    if length_pointer > 0:
        for label_id in label_sorted_by_length.keys():
            fill_count = math.ceil(label_sorted_by_length[label_id] /
                                   total_label_remain_length * length_pointer)
            fill_count = min(fill_count, label_remain_length_dict[label_id])
            offset_forward = fill_count
            if length_pointer - offset_forward <= 0 and length_pointer > 0:
                label_index_offset[label_id] += length_pointer
                length_pointer = 0
                break
            else:
                length_pointer -= offset_forward
                label_remain_length_dict[label_id] -= offset_forward
            label_index_offset[label_id] += offset_forward

    return label_index_offset


def label_skew_process(label_vocab, label_assignment, client_num, alpha,
                       data_length):
    """
    标签倾斜处理函数

    参数:
    -------------------------------------------------------------------
    label_vocab : dict，数据集的标签词汇表
    label_assignment : 1d list，标签列表，列表的索引与标签关联
    client_num : int，客户端数量
    alpha : float，每个客户端的相似性，alpha越大，每个客户端的数据越相似
    data_length : int，数据长度
    -------------------------------------------------------------------
    返回:
    ------------------------------------------------------------------
    partition_result : 2d array，每个客户端的分区索引列表
    ------------------------------------------------------------------
    """
    label_index_matrix = [[] for _ in label_vocab]
    label_proportion = []
    partition_result = [[] for _ in range(client_num)]
    client_length = 0

    # 打乱索引并计算每个标签在数据集中的比例
    for index, value in enumerate(label_vocab):
        label_location = np.where(label_assignment == value)[0]
        label_proportion.append(len(label_location) / data_length)
        np.random.shuffle(label_location)
        label_index_matrix[index].extend(label_location[:])

    # 计算每个分区客户端的大小
    label_index_tracker = np.zeros(len(label_vocab), dtype=int)
    total_index = data_length
    each_client_index_length = int(total_index / client_num)
    client_dir_dis = np.array([alpha * l for l in label_proportion])

    # 对于每个客户端计算分配给它的每个标签的长度
    for client_id in range(len(partition_result)):
        each_client_partition_result = partition_result[client_id]
        proportions = np.random.dirichlet(client_dir_dis)
        client_length = min(each_client_index_length, total_index)
        if total_index < client_length * 2:
            client_length = total_index
        total_index -= client_length
        client_length_pointer = client_length
        
        for label_id, _ in enumerate(label_vocab):
            offset = round(proportions[label_id] * client_length)
            if offset >= client_length_pointer:
                offset = client_length_pointer
                client_length_pointer = 0
            else:
                if label_id == (len(label_vocab) - 1):
                    offset = client_length_pointer
                client_length_pointer -= offset

            start = int(label_index_tracker[label_id])
            end = int(label_index_tracker[label_id] + offset)
            label_data_length = len(label_index_matrix[label_id])

            if end > label_data_length:
                each_client_partition_result.extend(
                    label_index_matrix[label_id][start:])
                label_index_tracker[label_id] = label_data_length
                label_index_offset = dynamic_batch_fill(
                    label_index_tracker, label_index_matrix,
                    end - label_data_length, label_id)
                for fill_label_id in label_index_offset.keys():
                    start = label_index_tracker[fill_label_id]
                    end = (label_index_tracker[fill_label_id] +
                           label_index_offset[fill_label_id])
                    each_client_partition_result.extend(
                        label_index_matrix[fill_label_id][start:end])
                    label_index_tracker[fill_label_id] = (
                        label_index_tracker[fill_label_id] +
                        label_index_offset[fill_label_id])
            else:
                each_client_partition_result.extend(
                    label_index_matrix[label_id][start:end])
                label_index_tracker[
                    label_id] = label_index_tracker[label_id] + offset

        if client_id == len(partition_result) - 1:
            for not_fillall_label_id in range(len(label_vocab)):
                if label_index_tracker[not_fillall_label_id] < len(
                        label_index_matrix[not_fillall_label_id]):
                    start = label_index_tracker[not_fillall_label_id]
                    each_client_partition_result.extend(
                        label_index_matrix[not_fillall_label_id][start:])
                    label_index_tracker[not_fillall_label_id] = len(
                        label_index_matrix[not_fillall_label_id])
        partition_result[client_id] = each_client_partition_result

    return partition_result
