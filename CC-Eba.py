import pandas as pd
import math
import numpy as np
import random
import time
import sys
# from Global_variable import *
import Global_variable

'''
abaa:预算平均分配算法
wqaa:基于课程单位质量报酬的加权分配算法
ga-abaa-100:平均分配预算的遗传算法迭代100代
ga-abaa-200:平均分配预算的遗传算法迭代200代
ga-wqaa-100:基于单位质量报酬加权分配预算的遗传算法迭代100代
ga-wqaa-200:基于单位质量报酬加权分配预算的遗传算法迭代200代
'''

'''输出一系列信息'''
def Print():

    print("均分预算:\n数据类型多样性指标 Diversity == ", Diversity)
    # print("总体单位数据质量报酬 Unit_Quality == ", Unit_Quality)
    print("课程支持数目 Task_num == ", Task_num)
    print("平均课件支持率 == ", Average_Courseware_Support_Rate)
    print("收集总数与要求总数的差的平方 == ", Difference)
    # print("时间盈余比率 == ", Average_Deadline)
    print("程序运行时间 time == ", t1)

    print("n ==", Global_variable.n)
    print("d ==", Global_variable.d)
    print("T[0] ==", Global_variable.T[0])
    print("max_num_type ==", Global_variable.max_num_type)

def init_var():#因为 N utype U 需要命令行读入，所以这里的变量需要重新变化

    Global_variable.dif_type = 0.2 * Global_variable.utype  # 针对每一个任务，提交的数据数目服从均匀分布，左右变动的界限
    # Global_variable.av_B = 350
    Global_variable.lrand = 0.2  # 随机数下界，如果生成的随机数大于该变量则进入T[i]
    Global_variable.B = Global_variable.av_B * Global_variable.N  # 随机生成总预算B
    Global_variable.vis = [0 for i in range(Global_variable.N)]  # 每一个课程是否可以留下，还是在数据清洗时直接扔掉（0代表留下，1代表扔掉）
    Global_variable.d = [0 for i in range(Global_variable.N)]  # 第i个任务要求的最晚结束时间
    Global_variable.n = [0 for i in range(Global_variable.N)]  # 第i个任务需要的数据数目（需要提交多少个数据）
    Global_variable.T = [[] for i in range(Global_variable.N)]  # 当前第i个任务是否需要某个类型j,把第i个任务需要的类型写入T[i]。  例如T[i].append(1); T[i].append(3);T[i].append(4);
    Global_variable.b = [0 for i in range(Global_variable.N)]  # 当前第i个任务的预算
    Global_variable.max_num_type = []  # 针对每个任务课程，类型出现次数的最大值


if __name__ == '__main__':
#############################################################################################################################
    Distribution_Choose = sys.argv[1]
    Global_variable.N = int(sys.argv[2])
    Global_variable.utype = int(sys.argv[3])
    Global_variable.U = int(sys.argv[4])
    Global_variable.av_B = int(sys.argv[5])

    init_var()  #初始化变量


    Global_variable.n = Global_variable.read_txt("n", 0)
    Global_variable.d = Global_variable.read_txt("d", 0)
    Global_variable.T = Global_variable.read_txt("T", 1)
    Global_variable.max_num_type = Global_variable.read_txt("max_num_type", 0)

    print("分布选择为:",Distribution_Choose, "预算B == ", Global_variable.B)
    print("N == ", Global_variable.N, " |T| == ", Global_variable.utype, " U == ", Global_variable.U)
    # Global_variable.Data_input(Global_variable.N,Global_variable.U,Global_variable.utype)

    s1 = time.time()
    Global_variable.Average_Allocation(Global_variable.N,Global_variable.U,Global_variable.utype)
    Diversity, Average_Courseware_Support_Rate, Difference, Task_num = Global_variable.Greedy(Global_variable.N,Global_variable.U,Global_variable.utype)
    s2 = time.time()
    t1 = s2 - s1

    Print() # 输出一系列信息