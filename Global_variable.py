import pandas as pd
import math
import numpy as np
import random
import time
import sys

utype = 0  # 类型数目的最大值的上界
U = 0  # 用户数目
N = 0  # 随机生成任务的数目N

dif_type = 0.2 * utype  # 针对每一个任务，提交的数据数目服从均匀分布，左右变动的界限
av_B = 80
lrand = 0.2  # 随机数下界，如果生成的随机数大于该变量则进入T[i]
B = av_B * N  # 随机生成总预算B
vis = [0 for i in range(N)]  # 每一个课程是否可以留下，还是在数据清洗时直接扔掉（0代表留下，1代表扔掉）
d = [0 for i in range(N)]  # 第i个任务要求的最晚结束时间
n = [0 for i in range(N)]  # 第i个任务需要的数据数目（需要提交多少个数据）
T = [[] for i in range(N)]  # 当前第i个任务是否需要某个类型j,把第i个任务需要的类型写入T[i]。  例如T[i].append(1); T[i].append(3);T[i].append(4);
b = [0 for i in range(N)]  # 当前第i个任务的预算
max_num_type = []  # 针对每个任务课程，类型出现次数的最大值
beta = 0



'''生成任意的三种分布'''
def Arbitrary_Distribution(in_num,flag):
    suoshuleixing = []
    # in_num = 50*utype
    if flag == '1':
        para = 0.3
        mu = 0
        sigma = 1
        yipusilong = 5 * sigma
        for t in range(in_num):
            r = random.random()
            # print(r)
            if r < para:
                tmp = int(np.random.randint(low=0, high=utype, size=1))
                suoshuleixing.append(tmp)
            elif r >= para and r <= 2 * para:
                tmp = np.random.normal(mu, sigma, 1)
                while tmp > yipusilong or tmp < -yipusilong:
                    tmp = np.random.normal(mu, sigma, 1)
                tmp = int((utype - 1) * (tmp - mu + yipusilong) / (10 * sigma))
                suoshuleixing.append(tmp)
            else:
                Lamda = 1.0 / (mu + 1)
                y = random.random()
                x = -math.log(y) / Lamda
                x = (utype - 1) * (x - 0) / (mu + 1 + yipusilong)  # 最后的类型值
                x = int(x)
                suoshuleixing.append(x)
    elif flag == '2':
        para = 0.35
        mu = 0
        sigma = 1
        yipusilong = 5 * sigma
        for t in range(in_num):
            r = random.random()
            # print(r)
            if r < para:
                tmp = int(np.random.randint(low=0, high=utype, size=1))
                suoshuleixing.append(tmp)
            elif r >= para and r <= 2 * para:
                tmp = np.random.normal(mu, sigma, 1)
                while tmp > yipusilong or tmp < -yipusilong:
                    tmp = np.random.normal(mu, sigma, 1)
                tmp = int((utype - 1) * (tmp - mu + yipusilong) / (10 * sigma))
                suoshuleixing.append(tmp)
            else:
                Lamda = 1.0 / (mu + 1)
                y = random.random()
                x = -math.log(y) / Lamda
                x = (utype - 1) * (x - 0) / (mu + 1 + yipusilong)  # 最后的类型值
                x = int(x)
                suoshuleixing.append(x)
    elif flag=='3':
        para = 0.4
        mu = 0
        sigma = 1
        yipusilong = 5 * sigma
        for t in range(in_num):
            r = random.random()
            # print(r)
            if r < para:
                tmp = int(np.random.randint(low=0, high=utype, size=1))
                suoshuleixing.append(tmp)
            elif r >= para and r <= 2 * para:
                tmp = np.random.normal(mu, sigma, 1)
                while tmp > yipusilong or tmp < -yipusilong:
                    tmp = np.random.normal(mu, sigma, 1)
                tmp = int((utype - 1) * (tmp - mu + yipusilong) / (10 * sigma))
                suoshuleixing.append(tmp)
            else:
                Lamda = 1.0 / (mu + 1)
                y = random.random()
                x = -math.log(y) / Lamda
                x = (utype - 1) * (x - 0) / (mu + 1 + yipusilong)  # 最后的类型值
                x = int(x)
                suoshuleixing.append(x)
    return suoshuleixing

'''生成三种质量的分布：低、中、高'''
def Quality_Distribution(in_num):
    suoshuleixing = []
    # in_num = 50*utype
    para = random.random()
    mu = 0
    sigma = 1
    yipusilong = 5 * sigma

    if para < 0.33:  # [1,10]  [11,30]
        for i in range(in_num):
            r = random.random()
            if r < 0.8:
                x = int(np.random.randint(low=0, high=10 + 1, size=1))
            else:
                x = int(np.random.randint(low=11, high=30 + 1, size=1))
            suoshuleixing.append(x)
    elif para < 0.66:  # [11,22]
        for i in range(in_num):
            r = random.random()
            if r < 0.8:
                x = int(np.random.randint(low=11, high=22 + 1, size=1))
            else:
                rr = random.random()
                if rr < 0.5:
                    x = int(np.random.randint(low=1, high=10 + 1, size=1))
                else:
                    x = int(np.random.randint(low=23, high=30 + 1, size=1))
            suoshuleixing.append(x)
    else:  # [20,30]
        for i in range(in_num):
            r = random.random()
            if r < 0.8:
                x = int(np.random.randint(low=20, high=30 + 1, size=1))
            else:
                x = int(np.random.randint(low=1, high=20 + 1, size=1))
            suoshuleixing.append(x)
    return suoshuleixing

''' 生成一系列用户数据，即哪些用户针对哪个任务提交了多少个哪种类型的数据'''
def Data_input(N,U,utype,flag):

    # in_num = np.random.randint(low=utype-dif_type, high=utype+dif_type, size=N)# 欲参与这N个任务中的每个任务的提交数据的次数

    # 30*utype
    in_num = [20*utype for i in range(N)]

    for j in range(N): #一个任务会有多个类型！！！
        user_id = np.random.randint(low=0, high=U, size=in_num[j])# 用户序号

        last_time = np.random.randint(low=5, high=30 + 1, size=in_num[j])# 最晚开始时间

        daijia = np.random.randint(low=3, high=15 + 1, size=in_num[j])# 代价

        baochou = np.random.randint(low=12, high=30 + 1, size=in_num[j])# 报酬

        suoshuleixing = []
        if flag =='U': #根据输入参数判断进行哪一个分布
            suoshuleixing = np.random.randint(low=0, high=utype, size=in_num[j])# 所属类型
        elif flag == 'N':
            suoshuleixing = np.random.normal(0, 1, in_num[j])  # 均值为mu，均方差为sigma，生成数目为num
            for t in range(in_num[j]):
                while suoshuleixing[t] > 5 or suoshuleixing[t] < -5:
                    suoshuleixing[t] = np.random.normal(0, 1, 1)  # 均值为mu，均方差为sigma，生成数目为num
                suoshuleixing[t] = int((suoshuleixing[t] + 5) * (utype - 1) / 10)
        elif flag == 'E':
            suoshuleixing = []
            for i in range(in_num[j]):
                y = random.random()
                x = -math.log(y)
                x = int(x * (N - 1) / (1 + 5))  # 最后的类型值
                suoshuleixing.append(x)
        else:
            suoshuleixing = Arbitrary_Distribution(in_num[j],flag)

        now = 0  #针对每个任务课程，计算类型出现次数的最大值
        TTT = list(suoshuleixing)
        for k in TTT:
            tmp = TTT.count(k)
            now = max(now,tmp)
        max_num_type.append(now)

        # shujuzhiliang = np.random.randint(low=2, high=30 + 1, size=in_num[j])# 数据质量
        shujuzhiliang = Quality_Distribution(in_num[j])

        data = {'用户序号':user_id,
                '最晚开始时间':last_time,
                '代价':daijia,
                '报酬':baochou,
                '所属类型':suoshuleixing,
                '数据质量':shujuzhiliang
                }
        frame1 = pd.DataFrame(data)
        frame1.to_excel('sci_data_'+str(j)+'.xlsx', sheet_name='任务' + str(j), index=None)

    '''生成N个任务的数据，即某个任务的最晚结束时间，需要的数据数目，需要哪些类型'''
    for i in range(N):
        d[i] = random.randint(10, 40)
        n[i] = random.randint(10, 40)

        for j in range(utype):
            Rand = random.random()  #生成[0,1)之间的随机小数
            if(Rand>lrand):
                T[i].append(j)  #第i个任务需要类型j

    write_to_file(n,0,"n")      # 写入文件
    write_to_file(d,0,"d")
    write_to_file(T,1,"T")
    write_to_file(max_num_type,0,"max_num_type")
    # n = read_txt("n",0)
    # d = read_txt("d",0)
    # T = read_txt("T",1)

'''读取txt文件'''
def read_txt(str, flag):
    if flag == 0: # n,d，max_num_type  只有一行的
        for line in open(str + '.txt', 'r'):
            line = list(map(float, line[1:-1].split(', ')))
            return line
    elif flag == 1: #T 有多行的
        ans = []
        for line in open(str + '.txt', 'r'):
            # line = line[1:-1].split(', ')
            ans.append(list(map(float, line[1:-2].split(', '))))
        return ans

'''写入txt文件'''
def write_to_file(Str, flag, name):
    file = open(name + ".txt", 'w', encoding='utf-8')  # 'w'新建只写
    if flag == 1:
        for len in Str:
            file.write(str(len) + '\n')
    else:
        file.write(str(Str))
    file.close()

'''对预算进行平均分配'''
def Average_Allocation(N,U,utype):
    # print("均分预算 B ==", B)
    for i in range(N):
        b[i] = B / N
        # print('b[', i, ']==', b[i])

'''对预算进行加权分配(加权标准为:单位质量报酬)，需要提前有数据清洗，标准为质量之和的平均值'''
def Weighted_Allocation(N,U,utype):
    Q_R = read_txt("Q_R", 0)
    Q_R = Q_R[0:N]
    now = []
    for i in range(N):
        now.append([i, Q_R[i]])  # [课程编号，课程的 数据质量/报酬 和]
    now = sorted(now, key=lambda x: x[1])

    # Quality = [0 for i in range(N)]
    # for i in range(N):
    #     data = pd.read_excel("sci_data_" + str(i) + ".xlsx")
    #     List = data['数据质量']
    #     Quality[i] = sum(List)
    # now = []
    # for i in range(N):
    #     now.append([i, Q[i]])  # [课程编号，课程的数据质量和]
    # now = sorted(now, key=lambda x: x[1])

    # Sum = 0
    for i in range(int(beta * N)):
        Id = now[i][0]
        vis[Id] = 1  # 过滤掉
        Q_R[Id] = 0
    Sum = sum(Q_R)
    # Quality = [0 for i in range(N)]
    # print("加权分配预算 B==", B)
    # for i in range(N):
    #     data = pd.read_excel("sci_data_" + str(i) + ".xlsx")
    #     List = data['数据质量'] / data['报酬']  # 单位质量报酬
    #     Quality[i] = sum(List)
    # Sum = sum(Quality)
    for i in range(N):
        b[i] = int(B * Q_R[i] / Sum)
        # print('b[', i, ']==', b[i])

'''按类型基于单位质量报酬排序的贪心算法'''
def Greedy(N,U,utype):
    Maximize_leixing = 0  # 类型多样性尽可能大
    Difference = 0  # 收集总数与要求总数的差的平分开根号
    Average_Courseware_Support_Rate = 0  # 平均课件支持率

    Maximize_R = 0  # 整体的单位质量报酬尽可能大
    Task_num = 0  # 支持任务数
    Minimize = 0  # 尽可能满足课件数目要求，该指标废弃！
    Average_Deadline = 0  # 每个课程提交时间与截止时间的差值的求和的平均值，即平均截止时间
    filepath = 'sci_data_'

    m = [0 for i in range(N)]  # 欲参加第i个任务的用户数目
    M = [[] for i in range(N)]  # 当前第i个任务中最终选择的类型为j的数据数目
    E = [0 for i in range(N)]  # 任务的数据类型多样性指标
    X = [0 for i in range(N)]  # 是否对某个任务i提供预算支持
    P = [[] for i in range(N)]  # P[i, j]为任务i中，最终所选类型为j的数据数目与任务i要求数据数目的比值
    for i in range(N):
        if (vis[i] == 1):  # 该课程i已被清洗，不能选入
            continue
        data = pd.read_excel(filepath + str(i) + '.xlsx')

        data['单位质量报酬'] = data['数据质量'] / data['报酬']
        data.sort_values(by=['所属类型', '单位质量报酬'], axis=0, ascending=[True, False], inplace=True)  # 类型升序，单位质量报酬降序

        num_leixing = max(data['所属类型'])  # 类型的数目，从0开始
        m[i] = max(data['用户序号'])  # 欲参加第i个任务的用户数目,从0开始到 m[i]
        last_time = []
        budget = b[i]  # 当前任务的预算
        need_num = n[i]  # 第i个任务需要多少个数据
        x = [[0 for j in range(num_leixing + 1)] for k in range(m[i] + 1)]  # 系统是否选取用户某个类型为j的数据

        for j in range(num_leixing):
            M[i].append(0)
            P[i].append(0)

        circle_num = 0
        if (budget <= 0):
            continue
        Sum_M = 0  #在当然课程i下，选择的课件数目
        while True:
            circle_num += 1
            for j in range(num_leixing):  #循环次数 utype
                if j not in T[i]:  # 如果不需要该类型，则直接跳过！
                    continue
                now_data = data[data.所属类型 == j]  # 切片赋值，选出所有类型为j的行
                for k in now_data.index:
                    ID = data['用户序号'][k]
                    type = data['所属类型'][k]
                    if now_data['最晚开始时间'][k] <= d[i]:
                        if now_data['报酬'][k] <= budget:
                            if need_num > 0:
                                if x[ID][type] == 0:
                                    x[ID][type] = 1
                                    budget -= now_data['报酬'][k]
                                    M[i][type] += 1
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
            if (circle_num > min(100,max_num_type[i])):
                break
        if need_num<n[i]:  #确保不会有0
            Average_Deadline = Average_Deadline+ 1 - max(last_time)/d[i]

        # for j in range(m[i]):
        #     for k in range(num_leixing):
        #         if x[j][k] == 1:
        #             M[i][k] += 1
        #             X[i] = 1
        #             Sum_M += 1
        # print("Sum_M == ",Sum_M)
        if Sum_M > 0:
            Task_num += 1
            Minimize += math.fabs(n[i] - Sum_M)
            print(i,"当前的课件支持率为:",Sum_M/n[i],Sum_M,n[i])
            Average_Courseware_Support_Rate+=Sum_M/n[i]
            Difference += (Sum_M-n[i])*(Sum_M-n[i])
            # print("Minimize == ", Minimize)
            # print("Task_num == ",Task_num)
        for j in range(num_leixing):
            if (Sum_M == 0):
                break
            P[i][j] = M[i][j] / Sum_M
            if P[i][j] != 0:
                E[i] += -(P[i][j] * math.log(P[i][j], 2)) * M[i][j]
                # print(E[i])

        Maximize_leixing += X[i] * E[i]
        # print("M=",M)
        # print("P=",P)
        # print("E=",E)
        # print("X=",X)
    Average_Courseware_Support_Rate/=Task_num
    Average_Deadline/=Task_num
    # Difference = math.sqrt(Difference)
    return Maximize_leixing/Task_num, Average_Courseware_Support_Rate, Difference, Task_num

'''二分查找，返回最后一个大于等于target的值的位置，如果nums中元素均大于target（即不存在<=target的元素）
则返回nums的长度（即target如果要插入到nums中，应该插入的位置）'''
def lower_bound(nums, target):
    low, high = 0, len(nums)-1
    pos = len(nums)
    while low<high:
        mid = (low+high)/2
        mid = int(mid) #要取整
        if nums[mid] > target:
            low = mid+1
        else:#>=
            high = mid
            #pos = high
    if nums[low]<=target:
        pos = low
    return pos

'''编码函数'''
def encode(N):
    '''
    :param N: 课程数目,即二进制的位数
    :return: N位二进制数
    '''
    binary = []
    for i in range(N):
        rand = random.random()
        if (rand >= 0.8):
            binary.append(0)
        else:
            binary.append(1)
    return binary


# '''按类型基于单位质量报酬排序的贪心算法'''
# def Greedy(N,U,utype):
#     Maximize_leixing = 0  # 类型多样性尽可能大
#     Maximize_R = 0  # 整体的单位质量报酬尽可能大
#     Task_num = 0  # 支持任务数
#     Minimize = 0  # 尽可能满足课件数目要求
#     Average_Courseware_Support_Rate = 0 # 平均课件支持率
#     Average_Deadline = 0  # 每个课程提交时间与截止时间的差值的求和
#     filepath = 'sci_data_'
#
#     m = [0 for i in range(N)]  # 欲参加第i个任务的用户数目
#     M = [[] for i in range(N)]  # 当前第i个任务中最终选择的类型为j的数据数目
#     E = [0 for i in range(N)]  # 任务的数据类型多样性指标
#     X = [0 for i in range(N)]  # 是否对某个任务i提供预算支持
#     P = [[] for i in range(N)]  # P[i, j]为任务i中，最终所选类型为j的数据数目与任务i要求数据数目的比值
#     for i in range(N):
#         if (vis[i] == 1):  # 该课程i已被清洗，不能选入
#             continue
#         data = pd.read_excel(filepath + str(i) + '.xlsx')
#
#         data['单位质量报酬'] = data['数据质量'] / data['报酬']
#         data.sort_values(by=['所属类型', '单位质量报酬'], axis=0, ascending=[True, False], inplace=True)  # 类型升序，单位质量报酬降序
#
#         num_leixing = max(data['所属类型'])  # 类型的数目，从0开始
#         m[i] = max(data['用户序号'])  # 欲参加第i个任务的用户数目,从0开始到 m[i]
#         last_time = []
#         budget = b[i]  # 当前任务的预算
#         need_num = n[i]  # 第i个任务需要多少个数据
#         x = [[0 for j in range(num_leixing + 1)] for k in range(m[i] + 1)]  # 系统是否选取用户某个类型为j的数据
#
#         for j in range(num_leixing):
#             M[i].append(0)
#             P[i].append(0)
#
#         circle_num = 0
#         if (budget <= 0):
#             continue
#         while True:
#             circle_num += 1
#             for j in range(num_leixing):
#                 if j not in T[i]:  # 如果不需要该类型，则直接跳过！
#                     continue
#                 now_data = data[data.所属类型 == j]  # 切片赋值，选出所有类型为j的行
#                 for k in now_data.index:
#                     ID = data['用户序号'][k]
#                     type = data['所属类型'][k]
#                     if now_data['最晚开始时间'][k] <= d[i]:
#                         if now_data['报酬'][k] <= budget:
#                             if need_num > 0:
#                                 if x[ID][type] == 0:
#                                     x[ID][type] = 1
#                                     budget -= now_data['报酬'][k]
#                                     need_num -= 1
#                                     Maximize_R += now_data['单位质量报酬'][k]
#                                     last_time.append(now_data['最晚开始时间'][k])
#                                     break  # 轮流制选择类型
#                     # if now_data['最晚开始时间'][k] <= d[i] and now_data['报酬'][k] <= budget and need_num > 0 and x[ID][type] == 0:
#
#             if need_num <= 0:
#                 # print('x:\n', x)
#                 break
#             if (circle_num > min(100,max_num_type[i])):
#                 break
#         if need_num<n[i]:  #确保不会有0
#             Average_Deadline = Average_Deadline+max(last_time)
#         Sum_M = 0
#         for j in range(m[i]):
#             for k in range(num_leixing):
#                 if x[j][k] == 1:
#                     M[i][k] += 1
#                     X[i] = 1
#                     Sum_M += 1
#         # print("Sum_M == ",Sum_M)
#         if Sum_M > 0:
#             Task_num += 1
#             Minimize += math.fabs(n[i] - Sum_M)
#             # print("Minimize == ", Minimize)
#             # print("Task_num == ",Task_num)
#         for j in range(num_leixing):
#             if (Sum_M == 0):
#                 break
#             P[i][j] = M[i][j] / Sum_M
#             if P[i][j] != 0:
#                 E[i] += -(P[i][j] * math.log(P[i][j], 2))
#                 # print(E[i])
#
#         Maximize_leixing += X[i] * E[i]
#         # print("M=",M)
#         # print("P=",P)
#         # print("E=",E)
#         # print("X=",X)
#     return Maximize_leixing, Maximize_R, Task_num, Minimize, Average_Deadline
