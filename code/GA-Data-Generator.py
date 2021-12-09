import pandas as pd
import math
import numpy as np
import random
import time
import sys
# from Global_variable import *
import Global_variable


def init_var():#因为 N utype U 需要命令行读入，所以这里的变量需要重新变化
    Global_variable.dif_type = 0.2 * Global_variable.utype  # 针对每一个任务，提交的数据数目服从均匀分布，左右变动的界限
    # Global_variable.av_B = 170
    Global_variable.lrand = 0.2  # 随机数下界，如果生成的随机数大于该变量则进入T[i]
    Global_variable.B = Global_variable.av_B * Global_variable.N  # 随机生成总预算B
    Global_variable.vis = [0 for i in range(Global_variable.N)]  # 每一个课程是否可以留下，还是在数据清洗时直接扔掉（0代表留下，1代表扔掉）
    Global_variable.d = [0 for i in range(Global_variable.N)]  # 第i个任务要求的最晚结束时间
    Global_variable.n = [0 for i in range(Global_variable.N)]  # 第i个任务需要的数据数目（需要提交多少个数据）
    Global_variable.T = [[] for i in range(Global_variable.N)]  # 当前第i个任务是否需要某个类型j,把第i个任务需要的类型写入T[i]。  例如T[i].append(1); T[i].append(3);T[i].append(4);
    Global_variable.b = [0 for i in range(Global_variable.N)]  # 当前第i个任务的预算
    Global_variable.max_num_type = []  # 针对每个任务课程，类型出现次数的最大值

'''子函数，只考虑一个课程，而非N个'''
def son_Greedy(i,N):
    Maximize_leixing = 0  # 类型多样性尽可能大
    Difference = 0  # 收集总数与要求总数的差的平分开根号
    Average_Courseware_Support_Rate = 0  # 平均课件支持率

    Maximize_R = 0  # 整体的单位质量报酬尽可能大
    Task_num = 0  # 支持任务数
    Minimize = 0  # 尽可能满足课件数目要求，该指标废弃！

    Average_Deadline = 0  # 每个课程提交时间与截止时间的差值的求和的平均值，即平均截止时间
    filepath = 'sci_data_'

    m = [0 for i in range(N)]  # 欲参加第i个任务的用户数目
    M = []  # 当前第i个任务中最终选择的类型为j的数据数目
    E = [0 for i in range(N)]  # 任务的数据类型多样性指标
    X = [0 for i in range(N)]  # 是否对某个任务i提供预算支持
    P = []  # P[i, j]为任务i中，最终所选类型为j的数据数目与任务i要求数据数目的比值

    data = pd.read_excel(filepath + str(i) + '.xlsx')

    data['单位质量报酬'] = data['数据质量'] / data['报酬']
    data.sort_values(by=['所属类型', '单位质量报酬'], axis=0, ascending=[True, False], inplace=True)  # 类型升序，单位质量报酬降序

    num_leixing = max(data['所属类型'])  # 类型的数目，从0开始
    m[i] = max(data['用户序号'])  # 欲参加第i个任务的用户数目,从0开始到 m[i]
    last_time = []
    budget = Global_variable.b[i]  # 当前任务的预算
    need_num = Global_variable.n[i]  # 第i个任务需要多少个数据
    x = [[0 for j in range(num_leixing + 1)] for k in range(m[i] + 1)]  # 系统是否选取用户某个类型为j的数据

    for j in range(num_leixing):
        M.append(0)
        P.append(0)

    circle_num = 0
    if (budget <= 0):
        return 0
    Sum_M = 0  #在当然课程i下，选择的课件数目
    while True:
        circle_num += 1
        for j in range(num_leixing):  #循环次数 utype
            if j not in Global_variable.T[i]:  # 如果不需要该类型，则直接跳过！
                continue
            now_data = data[data.所属类型 == j]  # 切片赋值，选出所有类型为j的行
            for k in now_data.index:
                ID = data['用户序号'][k]
                type = data['所属类型'][k]
                if now_data['最晚开始时间'][k] <= Global_variable.d[i]:
                    if now_data['报酬'][k] <= budget:
                        if need_num > 0:
                            if x[ID][type] == 0:
                                x[ID][type] = 1
                                budget -= now_data['报酬'][k]
                                M[type] += 1
                                X[i] = 1
                                Sum_M += 1
                                need_num -= 1
                                Maximize_R += now_data['单位质量报酬'][k]
                                last_time.append(now_data['最晚开始时间'][k])
                                break  # 轮流制选择类型
                # if now_data['最晚开始时间'][k] <= d[i] and now_data['报酬'][k] <= budget and need_num > 0 and x[ID][type] == 0:

        if need_num <= 0:
            # print('x:\n', x)
            break
        if (circle_num > min(100,Global_variable.max_num_type[i])):
            break
    if need_num<Global_variable.n[i]:  #确保不会有0
        Average_Deadline = Average_Deadline+ 1 - max(last_time)/Global_variable.d[i]


    if Sum_M > 0:
        Task_num += 1
        Difference += (Sum_M - Global_variable.n[i]) * (Sum_M - Global_variable.n[i])
        Minimize += math.fabs(Global_variable.n[i] - Sum_M)
        # print(i,"当前的课件支持率为:",Sum_M/Global_variable.n[i],Sum_M,Global_variable.n[i])
        Average_Courseware_Support_Rate+=Sum_M/Global_variable.n[i]
        # print("Minimize == ", Minimize)
        # print("Task_num == ",Task_num)
    for j in range(num_leixing):
        if (Sum_M == 0):
            break
        P[j] = M[j] / Sum_M
        if P[j] != 0:
            E[i] += -(P[j] * math.log(P[j], 2)) * M[j]
                # print(E[i])

    Maximize_leixing += X[i] * E[i]
        # print("M=",M)
        # print("P=",P)
        # print("E=",E)
        # print("X=",X)
    # Average_Courseware_Support_Rate/=Task_num
    # Average_Deadline/=Task_num
    return Maximize_leixing, Average_Courseware_Support_Rate, Difference, Task_num

'''按照mp的N种预算分配方式，生成5种指标的txt文件 #以及 Q(课程质量)、Q/R(单位质量报酬)的2个txt文件#'''
def Allocate_Budget():
    Maximize_leixing = [[] for i in range(len(mp))]
    # Maximize_R = [[] for i in range(len(mp))]
    Task_num = [[] for i in range(len(mp))]
    Average_Courseware_Support_Rate = [[] for i in range(len(mp))]
    # Average_Deadline = [[] for i in range(len(mp))]
    Difference = [[] for i in range(len(mp))]

    tmp = [mp[0],mp[len(mp)-1]]
    a = [[],[]]
    b = [[],[]]
    c = [[],[]]
    d = [[],[]]
    # e = [[],[]]
    s = time.time()
    for i in range(2): # 将这个N*N表格的计算简化，只需要算2次，然后把所有的结果都搞为这个两个数的均值
        for j in range(Global_variable.N):
            Global_variable.b[j] = Global_variable.B*tmp[i]
            # print(j,Global_variable.b[j])
        for j in range(Global_variable.N):
            aa,bb,cc,dd = son_Greedy(j,Global_variable.N)
            print(i, j, aa, bb, cc, dd)
            a[i].append(aa)
            b[i].append(bb)
            c[i].append(cc)
            d[i].append(dd)
            # e[i].append(ee)
    print("到算完2*N个son_Greedy花费的时间为",time.time()-s)
    for i in range(len(mp)):
        for j in range(Global_variable.N):
            Maximize_leixing[i].append((a[0][j]+a[1][j])/2)
            # Maximize_R[i].append((b[0][j]+b[1][j])/2)
            Task_num[i].append((d[0][j]+d[1][j])/2)
            Average_Courseware_Support_Rate[i].append((b[0][j]+b[1][j])/2)
            Difference[i].append((c[0][j]+c[1][j])/2)
            # Average_Deadline[i].append((e[0][j]+e[1][j])/2)
    # for i in range(len(mp)):  #横坐标代表预算，纵坐标代表课程下标
    #     for j in range(Global_variable.N):
    #         Global_variable.b[j] = Global_variable.B*mp[i]
    #         print(j,Global_variable.b[j])
    #     for j in range(Global_variable.N):
    #         a,b,c,d,e = son_Greedy(j,Global_variable.N)
    #         Maximize_leixing[i].append(a)
    #         Maximize_R[i].append(b)
    #         Task_num[i].append(c)
    #         Average_Courseware_Support_Rate[i].append(d)
    #         Average_Deadline[i].append(e)
    s = time.time()
    Global_variable.write_to_file(Maximize_leixing, 1, "Maximize_leixing")
    # Global_variable.write_to_file(Maximize_R, 1, "Maximize_R")
    Global_variable.write_to_file(Task_num, 1, "Task_num")
    Global_variable.write_to_file(Average_Courseware_Support_Rate, 1, "Average_Courseware_Support_Rate")
    # Global_variable.write_to_file(Average_Deadline, 1, "Average_Deadline")
    Global_variable.write_to_file(Difference, 1, "Difference")
    print("write_to_file四个文件需要的时间为:",time.time()-s)
    s = time.time()
    Q_R = [0 for j in range(Global_variable.N)]
    for j in range(Global_variable.N):
        data = pd.read_excel("sci_data_" + str(j) + ".xlsx")
        List = data['数据质量'] / data['报酬']  # 单位质量报酬
        Q_R[j] = sum(List)
    Global_variable.write_to_file(Q_R, 0, "Q_R")
    print("write_to_file Q_R需要的时间为:",time.time()-s)


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
#############################################################################################################################

    # mp = [1,0.9,0.8,0.7,0.6]
    # for i in range(2, Global_variable.N + 1):
    #     mp.append(1.0 / i)
    mp = [1, 1/Global_variable.N]
    # print(mp[2])
    print("分布选择为:", Distribution_Choose, "预算B == ", Global_variable.B)
    print("N == ", Global_variable.N, " |T| == ", Global_variable.utype, " U == ", Global_variable.U)
    # print(lower_bound(mp,0.0125))
    s = time.time()
    Allocate_Budget()
    print("运行完ga-Data-Generator总共花费的时间为：",time.time()-s)


    # # Global_variable.Data_input(Global_variable.N,Global_variable.U,Global_variable.utype)
    #
    # s1 = time.time()
    # Global_variable.Weighted_Allocation(Global_variable.N,Global_variable.U,Global_variable.utype)
    # Diversity, Unit_Quality, Task_num, Average_Courseware_Support_Rate, Average_Deadline = Global_variable.Greedy(Global_variable.N,Global_variable.U,Global_variable.utype)
    # s2 = time.time()
    # t1 = s2 - s1

    # Print() # 输出一系列信息