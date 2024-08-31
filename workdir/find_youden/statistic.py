import pandas as pd
import statistic_util

df_train_1 = pd.read_csv('workdir/find_youden/csvdata/20240219_result_ship_train_retina_asdd.csv')
df_train_2 = pd.read_csv('workdir/find_youden/csvdata/20240219_result_noship_train_retina_asdd.csv')

df_test_1 = pd.read_csv('workdir/find_youden/csvdata/20240219_result_ship_test_retina_asdd.csv')
df_test_2 = pd.read_csv('workdir/find_youden/csvdata/20240219_result_noship_test_retina_asdd.csv')

# statistic_util.original(df_train_1, df_train_2)
# statistic_util.method2(df_train_1, df_train_2, 0.5)     # 只在第四层过滤, 阈值0.5

threshold1, threshold2, threshold3, threshold4 = statistic_util.find_optimal_threshold(df_train_1, df_train_2, 0.98)
statistic_util.method3(df_test_1, df_test_2, threshold1, threshold2, threshold3, threshold4)   # 层层联合过滤

# retina asdd lianhe
# w = 0.5  : 0.659 0.709 0.799 0.716  |    49.2  73.5  51.2  38.2  68.0  53.3   |    25.1
# w = 0.6  : 0.622 0.682 0.773 0.656  |    53.3  79.1  55.8  40.1  74.9  68.6   |    22.9
# w = 0.7  : 0.602 0.648 0.725 0.62   |    55.1  81.6  57.7  40.7  78.3  75.8   |    21.6
# w = 0.8  : 0.579 0.623 0.674 0.58   |    56.3  83.3  58.9  41.3  80.2  81.4   |    20.5
# w = 0.9  : 0.548 0.591 0.616 0.532  |    57.1  84.3  59.8  41.6  81.5  83.1   |    19.3
# w = 0.95 : 0.524 0.556 0.554 0.46   |    57.6  85.0  60.4  41.8  82.4  85.7   |    18.3
# w = 0.96 : 0.515 0.534 0.548 0.46   |    57.7  85.1  60.4  41.8  82.5  85.7   |    18.0
# w = 0.97 : 0.51  0.51  0.527 0.417  |    57.7  85.1  60.5  41.8  82.6  85.7   |    17.7
# w = 0.98 : 0.489 0.51  0.514 0.417  |    57.7  85.1  60.5  41.8  82.6  85.7   |    17.7
# w = 0.99 : 0.473 0.51  0.49  0.417  |    57.7  85.2  60.5  41.8  82.7  85.7   |    17.6
# w = 1.0 :  0.29  0.353 0.329 0.177  |    57.7  85.2  60.5  41.8  82.8  86.2   |    16.0

np_x = [25.1, 22.9, 21.6, 20.5, 19.3, 18.3, 18.0, 17.7, 17.7, 17.6, 16.0, 12.6]
np_y = [73.5, 79.1, 81.6, 83.3, 84.3, 85.0, 85.1, 85.1, 85.1, 85.2, 85.2, 85.2]

np_x_baseline = [12.8]
np_y_baseline = [83.1]


statistic_util.draw_precision_speed_curve(np_x, np_y, np_x_baseline, np_y_baseline)