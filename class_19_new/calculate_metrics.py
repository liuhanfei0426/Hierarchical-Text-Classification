import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, f1_score

def calculate_metrics(true_labels, predicted_labels):
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    return {
        'micro': {
            'precision': precision_micro,
            'recall': recall_micro,
            'f1': f1_micro
        },
        'macro': {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        },
        'weighted': {
            'precision': precision_weighted,
            'recall': recall_weighted,
            'f1': f1_weighted
        }
    }

def calculate_evaluation_per_class(prediction, true_label):
    results = {}
    num_classes = true_label.shape[1]
    for class_index in range(num_classes):
        true_label_class = true_label[:, class_index]
        prediction_class = prediction[:, class_index]
        recall = recall_score(true_label_class, prediction_class, zero_division=0)
        precision = precision_score(true_label_class, prediction_class, zero_division=0)
        f1 = f1_score(true_label_class, prediction_class, zero_division=0)
        results[class_index] = {
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
    return results

# 统计4个大类的每一类的指标
def metric_big_per_class(prediction, true_label):
    results = {}
    size = (len(prediction), len(prediction[0]))
    split_list = [5, 10, 2, 2]
    total = 0

    for big_clss_id in range(len(split_list)):
        predict_pre_list, label_pre_list = [], []
        for sample_id in range(size[0]):
            pre_result, label_result = False, False
            for sub_class in range(total, total + split_list[big_clss_id]):
                pre_result = pre_result or prediction[sample_id][sub_class]
                label_result = label_result or true_label[sample_id][sub_class]

            predict_pre_list.append(int(pre_result))
            label_pre_list.append(int(label_result))

        total += split_list[big_clss_id]

        predict_pre_list = np.array(predict_pre_list, dtype=np.int32)
        label_pre_list = np.array(label_pre_list, dtype=np.int32)

        recall = recall_score(label_pre_list, predict_pre_list, zero_division=0)
        precision = precision_score(label_pre_list, predict_pre_list, zero_division=0)
        f1 = f1_score(label_pre_list, predict_pre_list, zero_division=0)

        results[big_clss_id] = {
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
    return results

# 统计4大类的总体结果
def metric_big_class(prediction, true_label):
    results = {}
    size = (len(prediction), len(prediction[0]))
    split_list = [5, 10, 2, 2]
    total = 0
    # 初始化用于存储拼接结果的矩阵
    prediction_4 = np.zeros((size[0], 0), dtype=np.int32)
    true_label_4 = np.zeros((size[0], 0), dtype=np.int32)

    for big_clss_id in range(len(split_list)):
        predict_pre_list, label_pre_list = [], []
        for sample_id in range(size[0]):
            pre_result, label_result = False, False
            for sub_class in range(total, total + split_list[big_clss_id]):
                pre_result = pre_result or prediction[sample_id][sub_class]
                label_result = label_result or true_label[sample_id][sub_class]

            predict_pre_list.append(int(pre_result))
            label_pre_list.append(int(label_result))

        total += split_list[big_clss_id]

        # 转换为NumPy数组，并将一维列表转换为一列
        predict_pre_list = np.array(predict_pre_list, dtype=np.int32).reshape(-1, 1)
        label_pre_list = np.array(label_pre_list, dtype=np.int32).reshape(-1, 1)

        # 按列拼接到现有矩阵
        prediction_4 = np.hstack((prediction_4, predict_pre_list))
        true_label_4 = np.hstack((true_label_4, label_pre_list))

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_label_4, prediction_4,
                                                                                 average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_label_4, prediction_4,
                                                                                 average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(true_label_4, prediction_4,
                                                                                          average='weighted')
    return {
        'micro': {
            'precision': precision_micro,
            'recall': recall_micro,
            'f1': f1_micro
        },
        'macro': {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        },
        'weighted': {
            'precision': precision_weighted,
            'recall': recall_weighted,
            'f1': f1_weighted
        }
    }


# 统计23类的总体的指标
def metric_23_class(prediction, true_label):
    results = {}
    size = (len(prediction), len(prediction[0]))
    split_list = [5, 10, 2, 2]
    total = 0
    # 初始化用于存储拼接结果的矩阵
    prediction_23 = np.array(prediction, dtype=np.int32)
    true_label_23 = np.array(true_label, dtype=np.int32)

    for big_clss_id in range(len(split_list)):
        predict_pre_list, label_pre_list = [], []
        for sample_id in range(size[0]):
            pre_result, label_result = False, False
            for sub_class in range(total, total + split_list[big_clss_id]):
                pre_result = pre_result or prediction[sample_id][sub_class]
                label_result = label_result or true_label[sample_id][sub_class]

            predict_pre_list.append(int(pre_result))
            label_pre_list.append(int(label_result))

        total += split_list[big_clss_id]

        # 转换为NumPy数组，并将一维列表转换为一列
        predict_pre_list = np.array(predict_pre_list, dtype=np.int32).reshape(-1, 1)
        label_pre_list = np.array(label_pre_list, dtype=np.int32).reshape(-1, 1)

        # 按列拼接到现有矩阵
        prediction_23 = np.hstack((prediction_23, predict_pre_list))
        true_label_23 = np.hstack((true_label_23, label_pre_list))


    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_label_23, prediction_23, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_label_23, prediction_23, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(true_label_23, prediction_23, average='weighted')
    return {
        'micro': {
            'precision': precision_micro,
            'recall': recall_micro,
            'f1': f1_micro
        },
        'macro': {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        },
        'weighted': {
            'precision': precision_weighted,
            'recall': recall_weighted,
            'f1': f1_weighted
        }
    }


if __name__ == "__main__":
    # 读取文件并提取前两列

    predict_data = np.loadtxt('predict_19.tsv', delimiter='\t', dtype=int)
    gt_data = np.loadtxt('gt_19.tsv', delimiter='\t', dtype=int)

    # 提取预测标签和真实标签
    # predict = predict_data[:, :2]  # 提取前二列
    # gt = gt_data[:, :2]  # 提取前二列
    predict = predict_data
    gt = gt_data

    print(predict_data)
    print(predict)

    # 计算指标
    metrics = calculate_metrics(gt, predict)
    print(f"19标签总体测试结果: \n")
    print("Micro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(metrics['micro']['precision']*100,
                                                                        metrics['micro']['recall']*100,
                                                                        metrics['micro']['f1']*100))
    print("Macro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(metrics['macro']['precision']*100,
                                                                        metrics['macro']['recall']*100,
                                                                        metrics['macro']['f1']*100))
    print("Weighted: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(metrics['weighted']['precision']*100,
                                                                           metrics['weighted']['recall']*100,
                                                                           metrics['weighted']['f1']*100))

    # 预测19个子类每一类的结果：
    per_class_result = calculate_evaluation_per_class(predict, gt)
    print("分别统计19子类每一类的指标:")
    for key, value in per_class_result.items():
        print(
            f"{key}: Precision: {value['precision']*100:.2f}, Recall: {value['recall']*100:.2f}, F1: {value['f1']*100:.2f}")

    # 统计4个大类的每一类的结果（不管子类预测的对不对）
    big_per_class_result = metric_big_per_class(predict, gt)
    print("\n分别统计4个大类的指标:（不管子类预测的对不对）\n")
    for key, value in big_per_class_result.items():
        print(
            f"{key}: Precision: {value['precision']*100:.2f}, Recall: {value['recall']*100:.2f}, F1: {value['f1']*100:.2f}")


    # 统计4个大类的总体结果：
    big_class_result = metric_big_class(predict, gt)
    print(f"\n4个大类的总体结果: \n")
    print("Micro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(big_class_result['micro']['precision'] * 100,
                                                                        big_class_result['micro']['recall'] * 100,
                                                                        big_class_result['micro']['f1'] * 100))
    print("Macro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(big_class_result['macro']['precision'] * 100,
                                                                        big_class_result['macro']['recall'] * 100,
                                                                        big_class_result['macro']['f1'] * 100))
    print("Weighted: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(big_class_result['weighted']['precision'] * 100,
                                                                           big_class_result['weighted']['recall'] * 100,
                                                                           big_class_result['weighted']['f1'] * 100))

    # 统计23标签的总体结果：
    metrics_23 = metric_23_class(predict, gt)
    print(f"\n23标签的总体测试结果: \n")
    print("Micro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(metrics_23['micro']['precision']*100,
                                                                        metrics_23['micro']['recall']*100,
                                                                        metrics_23['micro']['f1']*100))
    print("Macro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(metrics_23['macro']['precision']*100,
                                                                        metrics_23['macro']['recall']*100,
                                                                        metrics_23['macro']['f1']*100))
    print("Weighted: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(metrics_23['weighted']['precision']*100,
                                                                           metrics_23['weighted']['recall']*100,
                                                                           metrics_23['weighted']['f1']*100))


