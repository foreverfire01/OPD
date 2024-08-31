import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


# ROC
# def find_optimal_cutoff_roc(label, y_prob, above_recall = 0.997):
#     fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
#
#     # ---
#     filtered_indices = np.where(tpr > above_recall)
#     tpr = tpr[filtered_indices]
#     fpr = fpr[filtered_indices]
#     thresholds = thresholds[filtered_indices]
#     # ---
#
#     roc_auc = metrics.auc(fpr, tpr)
#
#     y = tpr - fpr
#     youden_index = np.argmax(y)  # Only the first occurrence is returned.
#     optimal_threshold = thresholds[youden_index]
#     optimal_point = [fpr[youden_index], tpr[youden_index]]
#     return fpr, tpr, roc_auc, optimal_threshold, optimal_point

def find_optimal_cutoff_roc(label, y_prob, w):
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)

    y = 2 * (w * tpr + (1-w) * (1 - fpr)) - 1
    youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = thresholds[youden_index]
    optimal_point = [fpr[youden_index], tpr[youden_index]]
    return fpr, tpr, roc_auc, optimal_threshold, optimal_point

def draw_threshold_roc(labels, scores, w):
    # labels = np.array([0, 0, 1, 1])
    # scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, roc_auc, optimal_th, optimal_point = find_optimal_cutoff_roc(labels, scores, w)

    plt.figure(1)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(optimal_point[0], optimal_point[1], marker='*', color='r')

    text_content = f'Best Threshold: {optimal_th:.2f}\n' \
                   f'TPR: {optimal_point[1]:.3f}\n' \
                   f'FPR: {optimal_point[0]:.3f}'
    plt.text(optimal_point[0], optimal_point[1] - 0.15, text_content)
    # plt.scatter(scores, labels, color='blue', s=0.5)
    # plt.title("ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


# PR
def find_optimal_cutoff_pr(label, y_prob):
    precision, recall, thresholds = metrics.precision_recall_curve(label, y_prob)
    pr_auc = metrics.auc(recall, precision)

    f1 = 2 * precision * recall / (precision + recall)
    f1_index = np.argmax(f1)  # Only the first occurrence is returned.
    optimal_threshold = thresholds[f1_index]
    optimal_point = [precision[f1_index], recall[f1_index]]
    return precision, recall, pr_auc, optimal_threshold, optimal_point


def test_threshold_pr(labels, scores):
    precision, recall, pr_auc, optimal_th, optimal_point = find_optimal_cutoff_pr(labels, scores)

    plt.figure(1)
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], f'Best Threshold:{optimal_th:.2f}')
    plt.scatter(scores, labels, color='blue')
    plt.title("Precision-Recall Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


# label = np.array([0, 0, 1, 1])
# y_prob = np.array([0.1, 0.4, 0.35, 0.8])
# test_threshold_roc(label, y_prob)
# test_threshold_pr(label, y_prob)