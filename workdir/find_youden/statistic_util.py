import pandas as pd
import find_optimal_cutoff as findthreshold
import sklearn



def original(df1 , df2):
    print("---原始数据每层情况，每层独立无关 阈值统一为0.5 > ")
    print("---      TP    FP    FN   TN")
    threshold = 0.5
    # stage 1
    TP = df1[df1.iloc[:, 1] - threshold> 0.00001]
    FP = df2[df2.iloc[:, 1] - threshold > 0.00001]

    number_TP = TP.shape[0]
    number_FP = FP.shape[0]
    number_FN = df1.shape[0] - TP.shape[0]
    number_TN = df2.shape[0] - FP.shape[0]

    print('stage 1: %d, %d, %d,   %d' %(number_TP, number_FP, number_FN, number_TN))

    # stage 2
    TP = df1[df1.iloc[:, 2] - threshold > 0.00001]
    FP = df2[df2.iloc[:, 2] - threshold > 0.00001]

    number_TP = TP.shape[0]
    number_FP = FP.shape[0]
    number_FN = df1.shape[0] - TP.shape[0]
    number_TN = df2.shape[0] - FP.shape[0]

    print('stage 2: %d, %d, %d,   %d' %(number_TP, number_FP, number_FN, number_TN))

    # stage 3
    TP = df1[df1.iloc[:, 3] - threshold > 0.00001]
    FP = df2[df2.iloc[:, 3] - threshold > 0.00001]

    number_TP = TP.shape[0]
    number_FP = FP.shape[0]
    number_FN = df1.shape[0] - TP.shape[0]
    number_TN = df2.shape[0] - FP.shape[0]

    print('stage 3: %d, %d, %d,   %d' %(number_TP, number_FP, number_FN, number_TN))

    # stage 4
    TP = df1[df1.iloc[:, 4] - threshold > 0.00001]
    FP = df2[df2.iloc[:, 4] - threshold > 0.00001]

    number_TP = TP.shape[0]
    number_FP = FP.shape[0]
    number_FN = df1.shape[0] - TP.shape[0]
    number_TN = df2.shape[0] - FP.shape[0]

    print('stage 4: %d, %d, %d,   %d' %(number_TP, number_FP, number_FN, number_TN))


def method2(df1 , df2, threshold4):
    print("---方法二，只在第四层过滤")
    print("---      TP    FP    FN   TN")
    threshold = threshold4

    # stage 4
    TP = df1[df1.iloc[:, 4] - threshold > 0.00001]
    FP = df2[df2.iloc[:, 4] - threshold > 0.00001]

    number_TP = TP.shape[0]
    number_FP = FP.shape[0]
    number_FN = df1.shape[0] - TP.shape[0]
    number_TN = df2.shape[0] - FP.shape[0]

    print('stage 4: %d, %d, %d,   %d' %(number_TP, number_FP, number_FN, number_TN))


def method3(df1, df2, threshold1, threshold2, threshold3, threshold4):
    print("---方法三，层层联合过滤, 用YD最佳阈值")
    print("---      TP    FP    FN   TN")

    # stage 1
    TP = df1[df1.iloc[:, 1] - threshold1 > 0.00001]
    FP = df2[df2.iloc[:, 1] - threshold1 > 0.00001]

    number_TP = TP.shape[0]
    number_FP = FP.shape[0]
    number_FN = df1.shape[0] - TP.shape[0]
    number_TN = df2.shape[0] - FP.shape[0]

    print('stage 1: %d, %d, %d,   %d' %(number_TP, number_FP, number_FN, number_TN))


    # stage 2
    TP = TP[TP.iloc[:, 2] - threshold2 > 0.00001]
    FP = FP[FP.iloc[:, 2] - threshold2 > 0.00001]

    number_TP = TP.shape[0]
    number_FP = FP.shape[0]
    number_FN = df1.shape[0] - TP.shape[0]
    number_TN = df2.shape[0] - FP.shape[0]

    print('stage 1: %d, %d, %d,   %d' %(number_TP, number_FP, number_FN, number_TN))


    # stage 3
    TP = TP[TP.iloc[:, 3] - threshold3 > 0.00001]
    FP = FP[FP.iloc[:, 3] - threshold3 > 0.00001]

    number_TP = TP.shape[0]
    number_FP = FP.shape[0]
    number_FN = df1.shape[0] - TP.shape[0]
    number_TN = df2.shape[0] - FP.shape[0]

    print('stage 3: %d, %d, %d,   %d' %(number_TP, number_FP, number_FN, number_TN))


    # stage 4
    TP = TP[TP.iloc[:, 4] - threshold4 > 0.00001]
    FP = FP[FP.iloc[:, 4] - threshold4 > 0.00001]

    number_TP = TP.shape[0]
    number_FP = FP.shape[0]
    number_FN = df1.shape[0] - TP.shape[0]
    number_TN = df2.shape[0] - FP.shape[0]

    print('stage 4: %d, %d, %d,   %d' %(number_TP, number_FP, number_FN, number_TN))


def find_optimal_threshold(df1, df2, w):
    labels = [1] * df1.shape[0] + [0] * df2.shape[0]

    scores1 = pd.concat([df1['col1'], df2['col1']], axis=0, ignore_index=True)
    _, _, _, threshold1, _ = findthreshold.find_optimal_cutoff_roc(labels, scores1, w)

    scores2 = pd.concat([df1['col2'], df2['col2']], axis=0, ignore_index=True)
    _, _, _, threshold2, _ = findthreshold.find_optimal_cutoff_roc(labels, scores2, w)

    scores3 = pd.concat([df1['col3'], df2['col3']], axis=0, ignore_index=True)
    _, _, _, threshold3, _ = findthreshold.find_optimal_cutoff_roc(labels, scores3, w)

    scores4 = pd.concat([df1['col4'], df2['col4']], axis=0, ignore_index=True)
    _, _, _, threshold4, _ = findthreshold.find_optimal_cutoff_roc(labels, scores4, w)

    print(w, ":", threshold1, threshold2, threshold3, threshold4)
    return threshold1, threshold2, threshold3, threshold4


def draw_precision_speed_curve(np_x, np_y, np_x_baseline, np_y_baseline):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import ConnectionPatch

    fig, ax = plt.subplots()
    plt.scatter(np_x_baseline, np_y_baseline, color='blue', marker='*', label='baseline', s=60)
    plt.scatter(np_x, np_y, color='red', label='ours', s=20)
    plt.plot(np_x, np_y, color='red')

    # plt.scatter(np_x_fenbu, np_y_fenbu, color='green', s=20)
    # plt.plot(np_x_fenbu, np_y_fenbu, color='green')

    plt.xlim(10, 26)
    plt.ylim(72, 92.5)
    plt.grid(True)
    plt.xlabel('FPS (img/s)')
    plt.ylabel('AP50')
    plt.legend()

    # 指定放大区域的范围
    x_start, x_end, y_start, y_end = 15.9, 19.4, 84.2, 85.5

    # 添加局部放大图
    local_x = [19.3, 18.3, 18.0, 17.7, 17.7, 17.6, 16.0]
    local_y = [84.3, 85.0, 85.1, 85.1, 85.1, 85.2, 85.2]
    axins = inset_axes(ax, width='30%', height='30%', loc='upper right')
    axins.plot(local_x, local_y, label='Zoomed In Data')

    # 设置放大区域的范围
    axins.set_xlim(x_start, x_end)
    axins.set_ylim(y_start, y_end)

    # 在放大图中绘制原始数据的部分
    axins.plot(local_x, local_y, 'o', markersize=3, color='red')

    # 添加连接放大图和原始图的箭头
    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.show()


def draw_firstlayers_roc(df1, df2, w, layer_index):
    labels = [1] * df1.shape[0] + [0] * df2.shape[0]
    layer = 'col' + str(layer_index)
    scores = pd.concat([df1[layer], df2[layer]], axis=0, ignore_index=True)
    _, _, _, threshold, _ = findthreshold.find_optimal_cutoff_roc(labels, scores, w)
    findthreshold.draw_threshold_roc(labels, scores, w)


def draw_alllayers_roc(df1, df2, w):
    labels = [1] * df1.shape[0] + [0] * df2.shape[0]

    scores1 = pd.concat([df1['col1'], df2['col1']], axis=0, ignore_index=True)
    scores2 = pd.concat([df1['col2'], df2['col2']], axis=0, ignore_index=True)
    scores3 = pd.concat([df1['col3'], df2['col3']], axis=0, ignore_index=True)
    scores4 = pd.concat([df1['col4'], df2['col4']], axis=0, ignore_index=True)

    fpr1, tpr1, roc_auc1, optimal_th1, optimal_point1 = findthreshold.find_optimal_cutoff_roc(labels, scores1, w)
    fpr2, tpr2, roc_auc2, optimal_th2, optimal_point2 = findthreshold.find_optimal_cutoff_roc(labels, scores2, w)
    fpr3, tpr3, roc_auc3, optimal_th3, optimal_point3 = findthreshold.find_optimal_cutoff_roc(labels, scores3, w)
    fpr4, tpr4, roc_auc4, optimal_th4, optimal_point4 = findthreshold.find_optimal_cutoff_roc(labels, scores4, w)

    import matplotlib.pyplot as plt

    plt.figure(1)

    plt.plot(fpr1, tpr1, label='stage 1', color=(77 / 255, 133 / 255, 189 / 255))
    text_content = f'stage 1: T= {optimal_th1:.2f}'
    plt.text(optimal_point1[0], optimal_point1[1] - 0.05, text_content)

    plt.plot(fpr2, tpr2, label='stage 2', color=(247 / 255, 144 / 255, 61 / 255))
    text_content = f'stage 2: T= {optimal_th2:.2f}'
    plt.text(optimal_point2[0], optimal_point2[1] - 0.12, text_content)
    plt.plot([optimal_point2[0], optimal_point2[0]], [optimal_point2[1], optimal_point2[1] - 0.09], linestyle=':', color='gray')

    plt.plot(fpr3, tpr3, label='stage 3', color='g')
    text_content = f'stage 3: T= {optimal_th3:.2f}'
    plt.text(optimal_point3[0], optimal_point3[1] - 0.05, text_content)

    plt.plot(fpr4, tpr4, label='stage 4', color=(210 / 255, 32 / 255, 39 / 255))
    text_content = f'stage 4: T= {optimal_th4:.2f}'
    plt.text(optimal_point4[0], optimal_point4[1] - 0.12, text_content)
    plt.plot([optimal_point4[0], optimal_point4[0]], [optimal_point4[1], optimal_point4[1] - 0.09], linestyle=':', color='gray')

    plt.plot(optimal_point1[0], optimal_point1[1], marker='*', markersize=10, color=(77 / 255, 133 / 255, 189 / 255))
    plt.plot(optimal_point2[0], optimal_point2[1], marker='*', markersize=10, color=(247 / 255, 144 / 255, 61 / 255))
    plt.plot(optimal_point3[0], optimal_point3[1], marker='*', markersize=10, color='g')
    plt.plot(optimal_point4[0], optimal_point4[1], marker='*', markersize=10, color=(210 / 255, 32 / 255, 39 / 255))

    plt.plot([0, 1], [0, 1], linestyle="--", color=(250 / 255, 192 / 255, 15 / 255))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()





# df_test_1 = pd.read_csv('workdir/find_youden/csvdata/20240219_result_ship_test_retina_asdd.csv')
# df_test_2 = pd.read_csv('workdir/find_youden/csvdata/20240219_result_noship_test_retina_asdd.csv')

# draw_alllayers_roc(df_test_1, df_test_2, 0.96)





