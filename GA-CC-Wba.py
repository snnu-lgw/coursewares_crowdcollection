import numpy as np
import math
import random
import pandas as pd
import time
import Global_variable
import sys
'''
初始的总预算B需要给定
1.种群初始化:一串01变量(表示该课程是否选择)
2.适应度函数: fitness()
3.选择函数:  select()
4.交叉函数:  cross()
5.变异函数:  mutation()
'''

'''预处理函数，把后百分之5的或者10的课程直接pass掉'''
def Pre_fun(N, beta):
    now = []
    for i in range(N):
        now.append([i, Q_R[i]])  # [课程编号，课程的 数据质量/报酬 和]
    now = sorted(now, key=lambda x: x[1])

    for i in range(int(beta * N)):
        Id = now[i][0]
        vis[Id] = 1  # 默认永远为0，过滤掉
        Q_R[Id] = 0  # 相应的读取的Q_R的值也要变为0，否则后面加权分配的时候，这个pass就没有意义了！
        # print(Id, "被过滤掉了")

'''适应度函数:加权分配预算'''
def fitness2(N,M,unit):
    global cnt

    a = []
    b = []
    c = []
    d = []
    # e = []
    # Sum = sum(Q_R)

    cnt += 1
    for j in range(N):
        if (unit[j] == 0 or vis[j]==1):
            Global_variable.b[j] = 0
            continue
        Global_variable.b[j] = int(Global_variable.B * Q_R[j] * unit[j] / Sum_QR)
        # now_b = Global_variable.b[j]
        # if (now_b == 0):
        #     continue
        # if (now_b/Global_variable.B >= 0.50):
        #     index = 0
        # else:
        #     index = 1
        # # index = Global_variable.lower_bound(mp, now_b/Global_variable.B)  # 横坐标确定
        # if (index == len(mp)):
        #     index = -1
        index = 0
        a.append(Maximize_leixing[index][j])
        # b.append(Maximize_R[index][j])
        b.append(Average_Courseware_Support_Rate[index][j])
        c.append(Difference[index][j])
        d.append(Task_num[index][j])
    # return sum(a) / sum(d)  # N门课程的平均类型多样性
    return sum(a)/sum(d), sum(b) / sum(d), sum(c), sum(d)  # 除以 Task_num

def search_answer(N, M, population,flag):
    a = [0 for i in range(len(population))]  # M是种群个体数
    b = [0 for i in range(len(population))]
    c = [0 for i in range(len(population))]
    # d = [0 for i in range(M)]
    # e = [0 for i in range(M)]

    for i in range(M):
        if flag == 0:
            a[i] = fitness1(N, M, population[i])
        else:
            a[i], _, _, _ = fitness2(N, M, population[i])

    new_a = [1 for i in range(len(population))]
    # new_b = [1 for i in range(len(population))]
    # new_c = [1 for i in range(len(population))]
    # new_d = [1 for i in range(M)]
    # new_e = [1 for i in range(M)]

    if min(a) != max(a):
        new_a = [(a[i] - min(a)) / (max(a) - min(a)) for i in range(len(population))]
    # if max(b) != min(b):
    #     new_b = [(b[i] - min(b)) / (max(b) - min(b)) for i in range(len(population))]
    # if max(c) != min(c):
    #     new_c = [(c[i] - min(c)) / (max(c) - min(c)) for i in range(len(population))]
    # if max(d) != min(d):
    #     new_d = [(d[i] - min(d)) / (max(d) - min(d)) for i in range(M)]
    # if max(e) != min(e):
    #     new_e = [(e[i] - min(e)) / (max(e) - min(e)) for i in range(M)]
    # f = [new_a[i] + new_b[i] + new_c[i] + new_d[i] + new_e[i] for i in range(M)]
    f = [new_a[i] for i in range(len(population))]

    max_id = f.index(max(f))
    # print("max_id == ", max_id)
    # print("max_f == ", f[max_id])
    # print("max_x == ", new_x[max_id])
    # print("max_y == ", new_y[max_id])
    # print(population[max_id])

    if flag == 0:
        Diversity = fitness1(N, M, population[max_id])
        # Diversity, Average_Courseware_Support_Rate, Difference = fitness1(N, M, population[max_id])
    else:
        Diversity, Average_Courseware_Support_Rate, Difference, Task_num = fitness2(N, M, population[max_id])
    print("数据类型多样性指标 Diversity == ", Diversity)
    # print("总体单位数据质量报酬 Unit_Quality == ", Unit_Quality)
    print("课程支持数目 Task_num == ", Task_num)
    print("平均课件支持率 == ", Average_Courseware_Support_Rate)
    # print("时间盈余比率 == ", Average_Deadline)
    print("收集总数与要求总数的差的平方 == ", Difference)

    return Diversity  # , Average_Courseware_Support_Rate, Difference

'''选择函数'''
def select(N, M, population,flag):
    # 按照population的顺序存放其适应度
    a = [0 for i in range(len(population))] # M是种群个体数
    b = [0 for i in range(len(population))]
    c = [0 for i in range(len(population))]
    d = [0 for i in range(len(population))]
    e = [0 for i in range(len(population))]

    for i in range(len(population)):
        if flag==0:
            # print(len(population),i,population[i])
            # a[i], b[i], c[i] = fitness1(N, M, population[i])
            a[i] = fitness1(N, M, population[i])
        else :
            a[i],_,_ ,_= fitness2(N, M, population[i])

    new_a = [1 for i in range(len(population))]
    # new_b = [1 for i in range(len(population))]
    # new_c = [1 for i in range(len(population))]
    # new_d = [1 for i in range(M)]
    # new_e = [1 for i in range(M)]

    if min(a) != max(a):
        new_a = [(a[i]-min(a))/(max(a)-min(a)) for i in range(len(population))]
    # if max(b)!=min(b):
    #     new_b = [(b[i]-min(b))/(max(b)-min(b)) for i in range(len(population))]
    # if max(c) != min(c):
    #     new_c = [(c[i] - min(c)) / (max(c) - min(c)) for i in range(len(population))]
    # if max(d) != min(d):
    #     new_d = [(d[i] - min(d)) / (max(d) - min(d)) for i in range(M)]
    # if max(e) != min(e):
    #     new_e = [(e[i] - min(e)) / (max(e) - min(e)) for i in range(M)]

    # all_fitness = [new_a[i] + new_b[i] + new_c[i] + new_d[i] + new_e[i] for i in range(M)]
    all_fitness = [new_a[i] for i in range(len(population))]
    # all_fitness = [new_a[i] for i in range(M)]
    # ID = [i for i in range(len(population))]
    # print("len_population == ",len(population))
    mydict = dict(zip(all_fitness,population))
    new_dict = sorted(mydict.items(), key=lambda x: x[0], reverse=True)  # 按照值类型降序排列
    # print("len_new_dict == ",len(new_dict))
    new_p = list(dict(new_dict).values())

    population = new_p[0:M-1]

    # all_fitness = fitness(N,M,population)
    # sum_fitness = sum(all_fitness)

    # print("all_fitness == ", all_fitness)
    # print("sum_fitness == ", sum_fitness)

    # 以第一个个体为0号，计算每个个体轮盘开始的位置，position的位置和population是对应的
    # all_position = []
    # for i in range(M):
    #     all_position.append(sum(all_fitness[:i + 1]) / sum_fitness)

    # print("对应概率 all_position == ", all_position)
    next_population = population

    # for i in range(M):
    #     ret = random.random()
    #     while ret <= 0.0001:
    #         ret = random.random()
    #     for j in range(M):
    #         ret -= all_position[j]
    #         if ret <= 0:
    #             next_population.append(population[j])
    #             break
    return next_population

'''交叉函数'''
def cross(N, M, P_cross, population):
    # 计数器：交换次数
    # num = P_cross * M
    # count = 0
    # i = 0

    for j in range(0,int(0.15*M)):
        i = random.randint(0, len(population)-2)
        position = random.randrange(0, N-1)  # [0,N-1]
        # print(i + 1, position)

        tmp11 = population[i][:position]
        # tmp12 = population[i][position:]
        # tmp21 = population[i + 1][:position]

        tmp22 = population[i + 1][position:]
        # print(i+1, position, population[i+1], tmp22)
        # population[i] = tmp11 + tmp22
        # population[i + 1] = tmp21 + tmp12

        tmp1 = tmp11 + tmp22
        population.append(tmp1)
        # print(tmp1)
        # i += 2
        # count += 1
        # if (count > num):
        #     break
    # print("cross ", len(population))
    return population

'''变异函数'''
def mutation(N, M, P_mutation, population):

    for j in range(0,int(0.15*M)):

        i = random.randint(0, len(population)-1)
        tmp = population[i]
        for _ in range(int(P_mutation*M)):
            # ret = random.random()
            position = random.randrange(0, N)  # [0,N-1]
            while vis[position] == 1:
                position = random.randrange(0, N)  # [0,N-1]
            if tmp[position] == 0:
                tmp[position] = 1
            else:
                tmp[position] = 0
        population.append(tmp)

    return population
'''编码函数'''
def encode(N):
    '''
    :param N: 课程数目,即二进制的位数
    :return: N位二进制数
    '''
    binary = []
    for i in range(N):
        rand = random.random()
        if vis[i]==1:
            binary.append(0)
            continue

        if (rand >= 0.8):
            binary.append(0)
        else:
            binary.append(1)
    return binary

'''主函数'''
def main(N, M, T, P_cross, P_mutation,flag):
    all = []
    all_len = pow(10, 3)
    for i in range(all_len):
        all.append(encode(N))  # 生成all_len个随机01串,每个01串的长度为N，即判断N个课程是否选择
    # print(all)

    population = random.sample(all, M)  # 从all中随机选择M个01串
    # 迭代运行T次
    # MAX_Diversity = 0
    # MAX_Average_Courseware_Support_Rate = 0
    # MIN_Difference = 100000000
    for i in range(T):
        # 进行选择操作
        # print(len(population), population)
        if flag==0: # 均分预算
            population = select(N, M, population,0)
        else : # 加权分配预算
            population = select(N, M, population,1)
        # 进行交叉操作
        population = cross(N, M, P_cross, population)
        # 进行变异操作
        population = mutation(N, M, P_mutation, population)
        if (i == T - 1 and flag==0):
            print("\n遗传算法均分预算：")
            search_answer(N, M, population,0)
        if (i == T - 1 and flag == 1):
            print("\n遗传算法加权分配预算：")
            search_answer(N, M, population,1)
    #     if (flag==0):
    #         # print("\n遗传算法均分预算：")
    #         Diversity, Average_Courseware_Support_Rate, Difference = search_answer(N, M, population,0)
    #         MAX_Diversity = max(MAX_Diversity, Diversity)
    #         MAX_Average_Courseware_Support_Rate = max(MAX_Average_Courseware_Support_Rate, Average_Courseware_Support_Rate)
    #         MIN_Difference = min(MIN_Difference, Difference)
    #     if (flag == 1):
    #         # print("\n遗传算法加权分配预算：")
    #         Diversity, Average_Courseware_Support_Rate, Difference = search_answer(N, M, population,1)
    #         MAX_Diversity = max(MAX_Diversity, Diversity)
    #         MAX_Average_Courseware_Support_Rate = max(MAX_Average_Courseware_Support_Rate, Average_Courseware_Support_Rate)
    #         MIN_Difference = min(MIN_Difference, Difference)
    # if (flag == 0):
    #     print("\n遗传算法均分预算：")
    # else :
    #     print("\n遗传算法加权分配预算：")
    # print("数据类型多样性指标 Diversity == ", MAX_Diversity)
    # # print("总体单位数据质量报酬 Unit_Quality == ", Unit_Quality)
    # # print("课程支持率 Task_num/N == ", Task_num)
    # print("平均课件支持率 == ", MAX_Average_Courseware_Support_Rate)
    # # print("时间盈余比率 == ", Average_Deadline)
    # print("收集总数与要求总数的差的平方 == ", MIN_Difference)
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
    Age = int(sys.argv[5]) #迭代次数
    beta = float(sys.argv[6]) # beta值输入
    Global_variable.av_B = int(sys.argv[7])
    init_var()  # 初始化变量

    print("分布选择为:",Distribution_Choose, 'beta == ',beta, 'beta * N == ',beta*Global_variable.N,"预算B == ", Global_variable.B)
    print("N == ", Global_variable.N, " |T| == ", Global_variable.utype, " U == ", Global_variable.U)

    Global_variable.n = Global_variable.read_txt("n", 0)
    Global_variable.d = Global_variable.read_txt("d", 0)
    Global_variable.T = Global_variable.read_txt("T", 1)
    Global_variable.max_num_type = Global_variable.read_txt("max_num_type", 0)

    Maximize_leixing = Global_variable.read_txt("Maximize_leixing", 1)
    # Maximize_R = Global_variable.read_txt("Maximize_R", 1)
    Task_num = Global_variable.read_txt("Task_num", 1)
    Average_Courseware_Support_Rate = Global_variable.read_txt("Average_Courseware_Support_Rate", 1)
    Difference = Global_variable.read_txt("Difference", 1)
    # Average_Deadline = Global_variable.read_txt("Average_Deadline", 1)
    Q_R = Global_variable.read_txt("Q_R", 0)
    Sum_QR = sum(Q_R)
    vis = [0 for i in range(Global_variable.N)]
    mp = [1, 1 / Global_variable.N]
    # mp = [1,0.9,0.8,0.7,0.6]
    # for i in range(2, Global_variable.N + 1):
    #     mp.append(1.0 / i)

    cnt = 0

    Pre_fun(Global_variable.N, beta)
    # s1 = time.time()
    # main(Global_variable.N, 50, Age, 0.6, 0.01,0)
    # s2 = time.time()
    # t1 = s2 - s1
    # print("程序运行时间 time == ", t1)

    s1 = time.time()
    main(Global_variable.N, 200, Age, 0.6, 0.05, 1)
    s2 = time.time()
    t1 = s2 - s1
    print("程序运行时间 time == ", t1)

    print("n ==", Global_variable.n)
    print("d ==", Global_variable.d)
    print("T[0] ==", Global_variable.T[0])
    print("max_num_type ==", Global_variable.max_num_type)
    print("cnt == ",cnt)
