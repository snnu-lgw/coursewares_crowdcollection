import pandas as pd
import math
import numpy as np
import random
import time
import sys
import os
import Global_variable
from multiprocessing import Pool, Manager,Lock

def init_var():#因为 N utype U 需要命令行读入，所以这里的变量需要重新变化
    Global_variable.dif_type = 0.2 * Global_variable.utype  # 针对每一个任务，提交的数据数目服从均匀分布，左右变动的界限

    Global_variable.lrand = 0.2  # 随机数下界，如果生成的随机数大于该变量则进入T[i]

    Global_variable.d = [0 for i in range(Global_variable.N)]  # 第i个任务要求的最晚结束时间
    Global_variable.n = [0 for i in range(Global_variable.N)]  # 第i个任务需要的数据数目（需要提交多少个数据）
    Global_variable.T = [[] for i in range(Global_variable.N)]  # 当前第i个任务是否需要某个类型j,把第i个任务需要的类型写入T[i]。  例如T[i].append(1); T[i].append(3);T[i].append(4);
    Global_variable.max_num_type = []  # 针对每个任务课程，类型出现次数的最大值


# def main():
#     os.system("python abaa.py")

Distribution_Choose = sys.argv[1]
Global_variable.N = int(sys.argv[2])
Global_variable.utype = int(sys.argv[3])
Global_variable.U = int(sys.argv[4])

init_var()

print("分布选择为:",Distribution_Choose)

start = time.time()

# n_processes = 6
# pool = Pool(processes=n_processes)
# for k in range(n_processes):
#     pool.apply_async(Global_variable.Data_input, (Global_variable.N,Global_variable.U,Global_variable.utype,Distribution_Choose, k,Global_variable.N//n_processes,))  # 启动多进程
# pool.close()  # 使进程池不能添加新任务
# pool.join()  # 等待进程结束

Global_variable.Data_input(Global_variable.N,Global_variable.U,Global_variable.utype,Distribution_Choose)

print("########################\n生成数据花费的时间为：",time.time()-start,"\n########################")
