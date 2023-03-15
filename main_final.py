import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import pandas as pd
# coding: utf-8
import os
import json

def __find_peak__(acc,max_window):
    # Automatic multiscale-based peak detection (AMPD)自动多尺度峰值查找算法
    p_data = np.zeros_like(acc, dtype=np.int32)
    count = len(acc)
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for i in range(k, count - k):
            if acc[i] > acc[i - k] and acc[i] > acc[i + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    # print(max_window_length)
    if max_window_length>max_window:
        max_window_length = max_window-1
    # print(max_window_length)
    for k in range(1, max_window_length + 1):
        for i in range(k, count - k):
            if acc[i] > acc[i - k] and acc[i] > acc[i + k]:
                p_data[i] += 1
    return np.where(p_data == max_window_length)[0]
def Move_avg(data): #移动平均，长度为3
    data1 = []
    data1.append(data[0])
    for i in range(len(data)):
        if data[i]>2:
            data[i] = data[i]-2*np.pi
    for i in range(1,len(data)-1):
        data1.append((data[i-1] +data[i] +data[i+1])/3)
    data1.append(data[-1])
    return data1
def CDF(error,batch): #测试用
 # 计算误差累计函数
    lenaa = len(error)
    sorted_data = np.sort(error)
    error_50 = sorted_data[int(0.5*lenaa)]
    error_75 = sorted_data[int(0.75 * lenaa)]
    error_90 = sorted_data[int(0.9 * lenaa)]
    cumulative = np.cumsum(sorted_data)
    cumulative = cumulative / cumulative[-1]
    # 绘制误差累计曲线
    plt.figure()
    plt.plot(sorted_data, cumulative, label='CDF')
    # 添加标题和标签
    plt.title('Cumulative Distribution Function')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.savefig(str(batch)+'.png')
    plt.show()
    return (error_50, error_75, error_90)

class PDR: #处理数据的类
    def __init__(self,csv_position,csv_running,sample_batch,model): #读取数据并且预处理
        self.sample_batch = sample_batch
        self.accx = []
        self.accy = []
        self.accz = []
        self.acc = []
        self.gyroscopex = []
        self.gyroscopey = []
        self.gyroscopez = []
        self.stay = []
        self.timestamp = []
        # self.step_length = 0
        self.match_error = []
        self.min_len = 0
        self.step_frequency = []
        self.x_position = []
        self.y_position = []
        self.K = 0.45 #计算步长的系数
        self.x2_position = []
        self.y2_position = []
        self.model = int(model)
        self.index = []
        self.ki = 0.008   #互补滤波的积分参数
        self.Kp = 10    #互补滤波系数
        self.theta = []
        self.gama = []
        self.fai = []  #航向角
        self.cor = [] #C_E^b 姿态变换矩阵，从地球到物体
        self.inv_cor = [] # C_b^E 从物体到地球
        self.hangxiang = [0] #
        self.step_length = []
        self.x_true_position = [] #真实的位置
        self.y_true_position =[]
        self.x_2true_position = [] #真实的位置，但是跟小步的匹配
        self.y_2true_position = []
        # 要输出点的各个指标 方位角，步长
        self.direction = []
        self.length = []
        self.print_accx = []
        self.print_accy = []
        self.print_accz = []
        self.print_gyrx = []
        self.print_gyry = []
        self.print_gyrz = []
        self.index_begin = 0
        header = 0
        with open(csv_position) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                if header == 0 :
                    sample_batch_index = row.index('sample_batch')
                    x_position_index = row.index('x')
                    y_position_index = row.index('y')
                    header = 1
                    continue
                if row[sample_batch_index] == sample_batch:
                    self.x_position.append(eval(row[x_position_index]))
                    self.y_position.append(eval(row[y_position_index]))
        # 陀螺仪测量角速度的偏移，通过对稳定数据求平均得到，发现对两者有显著区别
        bias_gyx = 0.0288
        bias_gyy = -0.0293
        bias_gyz = -0.0451
        bias_gyx2 = 0.0403
        bias_gyy2 = -0.0293
        bias_gyz2 = -0.0251
        with open(csv_running) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            # header = next(csv_reader)        # 读取第一行每一列的标题
            begin = 0
            header = 0
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                if header == 0:
                    accx_index = row.index('accx')
                    accy_index = row.index('accy')
                    accz_index = row.index('accz')
                    gyrx_index = row.index('gyroscopex')
                    gyry_index = row.index('gyroscopey')
                    gyrz_index = row.index('gyroscopez')
                    timestamp_index = row.index('timestamp')
                    sample_batch_index = row.index('sample_batch')
                    stay_index = row.index('stay')
                    header = 1
                    continue
                if row[sample_batch_index] == sample_batch:
                    self.stay.append(eval(row[stay_index]))
                    if eval(row[stay_index]) == 1 and begin == 0:
                        continue
                    begin = 1
                    if self.model == 0: #因为一般的处理方法都是右手系，这里手表是左手系，所以输入的时候进行切换，以变成右手系。
                        self.accx.append(eval(row[accy_index]))
                        self.accy.append(eval(row[accx_index]))
                        self.accz.append(eval(row[accz_index]))
                        self.gyroscopex.append(eval(row[gyry_index])*0.08/180*np.pi-bias_gyy)
                        self.gyroscopey.append(eval(row[gyrx_index])*0.08/180*np.pi-bias_gyx)
                        self.gyroscopez.append(eval(row[gyrz_index])*0.08/180*np.pi-bias_gyz)
                    else:     #对于第二种模式手表佩戴情况，为了使用前一种模式的后续处理，将坐标系进行了替换，转换到和之前方向一样的情况。经过数据处理，感觉分辨率是不一样，因此乘的系数不一样。
                        self.accx.append(-1*eval(row[accx_index]))
                        self.accy.append(eval(row[accz_index]))
                        self.accz.append(eval(row[accy_index]))
                        self.gyroscopex.append(-1*eval(row[gyrx_index])*0.08/180*np.pi+bias_gyx2)
                        self.gyroscopey.append(eval(row[gyrz_index])*0.08/180*np.pi-bias_gyz2)
                        self.gyroscopez.append(eval(row[gyry_index])*0.08/180*np.pi-bias_gyy2)

                    self.timestamp.append(eval(row[timestamp_index]))
                    self.acc.append(np.sqrt(self.accx[-1]*self.accx[-1]+self.accy[-1]*self.accy[-1]+self.accz[-1]*self.accz[-1]))
        a = self.timestamp[0]
        for i in range(len(self.timestamp)):
            self.timestamp[i] -= a
            self.timestamp[i] = self.timestamp[i]/1000 #去掉时间戳无用的区域，并且换单位为秒。
        length = len(self.timestamp)
        # print(self.timestamp)
        # print(self.gyroscopez)
        self.accx = np.array(self.accx)
        self.accy = np.array(self.accy)
        self.accz = np.array(self.accz)
        self.acc = np.array(self.acc)
        self.gyroscopex = np.array(self.gyroscopex)
        self.gyroscopey = np.array(self.gyroscopey)
        self.gyroscopez = np.array(self.gyroscopez)

        #开始计算零偏的部分，得到后，直接设置，注释掉了，其中角速度gx，gy，通过第一种模式，基本不变平均得到，gz通过模式二有周期性得到。
        # accx_int = []
        # accx_int_tmp = 0
        # for i in range(length-1):
        #     accx_int_tmp +=(self.gyroscopey[i])*(self.timestamp[i+1]-self.timestamp[i])
        #     accx_int.append(accx_int_tmp)
        # accx_int.append(accx_int_tmp)
        # plt.scatter(self.timestamp, accx_int, color="red")
        # plt.show()
        # print('accxkofem')
        # print(accx_int[-2]/(self.timestamp[i]-self.timestamp[0]))
        # k = 0
        # for i in range(length):
        #     if self.stay[i]==1:
        #         k = k + 1
        #         bias_accx+=self.accx[i]
        #         bias_accy+=self.accy[i]
        #         bias_accz+=self.accz[i]
        #         bias_gyx+=self.gyroscopex[i]
        #         bias_gyy+=self.gyroscopey[i]
        #         bias_gyz+=self.gyroscopez[i]
        # bias_accx /= k
        # bias_accy /= k
        # bias_accz /= k
        # bias_gyx /= k
        # bias_gyy /= k
        # bias_gyz /= k
        Q = [1, 0, 0, 0] #初始化四元数方向
        Q = np.array(Q)
        [q0,q1,q2,q3]=Q
        [accex,accey,accez]=[0,0,0]
        jifen = 0
        for i in range(length-1):
            dt = self.timestamp[i+1]-self.timestamp[i]
            [ax, ay, az] = [self.accx[i],self.accy[i],self.accz[i]]
            [gx, gy, gz] = [self.gyroscopex[i], self.gyroscopey[i], self.gyroscopez[i]]
            [ax,ay,az] = Vsqrt([ax, ay, az])
            T = np.zeros(shape=(3, 3)) #求姿态矩阵
            T[0][0] = 1 - 2 * (q2 * q2 + q3 * q3)
            T[0][1] = 2 * (q1 * q2 + q0 * q3)
            T[0][2] = 2 * (q1 * q3 - q0 * q2)
            T[1][0] = 2 * (q1 * q2 - q0 * q3)
            T[1][1] = 1 - 2 * (q1 * q1 + q3 * q3)
            T[1][2] = 2 * (q2 * q3 + q0 * q1)
            T[2][0] = 2 * (q1 * q3 + q0 * q2)
            T[2][1] = 2 * (q2 * q3 - q0 * q1)
            T[2][2] = 1 - 2 * (q1 * q1 + q2 * q2)
            self.cor.append(T)
            self.inv_cor.append(T.T) #正交矩阵，转置就是逆，正好加进去
            b = np.array([0, 0, -1])
            #利用互补滤波的方法来通过重力方向和角速度积分的方法两个叠加来得到一个新的方向。
            [Vx, Vy, Vz] = np.dot(T, b)
            [ex, ey, ez] = [ay * Vz - az * Vy, az * Vx - ax * Vz, ax * Vy - ay * Vx]
            accex = accex + ex * self.ki
            accey = accey + ey * self.ki
            accez = accez + ez * self.ki
            gx = gx + self.Kp * ex + accex
            gy = gy + self.Kp * ey + accey
            gz = gz + self.Kp * ez + accez
            jifen = jifen  - (ax*gx+ay*gy+az*gz)*dt #投影到g坐标系的积分。
            # jifen = jifen  - (az*gz)*dt #投影到g坐标系的积分。
            self.hangxiang.append(jifen)
            W = np.array([gx * dt, gy * dt, gz * dt])
            L_mo = np.sqrt(np.sum(W ** 2))
            delta_theta = np.zeros(shape=(4, 4))
            delta_theta[0][1] = -gx
            delta_theta[1][0] = gx
            delta_theta[0][2] = -gy
            delta_theta[2][0] = gy
            delta_theta[0][3] = -gz
            delta_theta[3][0] = gz
            delta_theta[1][2] = gz
            delta_theta[2][1] = -gz
            delta_theta[1][3] = -gy
            delta_theta[3][1] = gy
            delta_theta[2][3] = gx
            delta_theta[3][2] = -gx
            Q = np.dot((1 - L_mo / 8 + L_mo * L_mo / 384) * np.identity(4) + (0.5 - L_mo / 48)*dt * delta_theta, Q)
            # Q = np.dot(1 * np.identity(4) + (0.5*dt) * delta_theta, Q)
            q0 = q0 - q1 * gx - q2 * gy - q3 * gz
            q1 = q1 + q0 * gx + q2 * gz - q3 * gy
            q2 = q2 + q0 * gy - q1 * gz + q3 * gx
            q3 = q3 + q0 * gz + q1 * gy - q2 * gx
            Q = Vsqrt(Q)
            [q0, q1, q2, q3] = Q
            g1 = 2 * (q1 * q3 - q0 * q2)
            g2 = 2 * (q2 * q3 + q0 * q1)
            g3 = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
            g4 = 2 * (q1 * q2 + q0 * q3)
            g5 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
            self.gama.append(-np.arcsin(g1))
            self.theta.append(np.arctan2(g2,g3))
            self.fai.append(np.arctan2(g4,g5))

        self.gama.append(-np.arcsin(g1))
        self.theta.append(np.arctan2(g2, g3))
        self.fai.append(np.arctan2(g4, g5))
        # self.hangxiang = Move_avg(self.hangxiang)
        # self.fai = Move_avg(self.fai)
        # plt.plot(self.timestamp, self.hangxiang, color="red")
        # plt.show()
    def Coordinate_system_rotation(self):
        for i in range(1, len(self.timestamp)):
            [self.accx[i], self.accy[i], self.accz[i]] = np.dot(self.inv_cor[i - 1],
                                                                [self.accx[i], self.accy[i], self.accz[i]])
        for i in range(1, len(self.timestamp)):
            [self.gyroscopex[i], self.gyroscopey[i], self.gyroscopez[i]] = np.dot(self.inv_cor[i - 1],
                                                                                  [self.gyroscopex[i],
                                                                                   self.gyroscopey[i],
                                                                                   self.gyroscopez[i]])
    def vis2(self): # 找到峰值，算步长，预测轨迹。
        L = []
        y = []
        self.p_max = __find_peak__(self.acc,3)
        self.p_min = __find_peak__(-1*self.acc,3)
    # plt.plot(self.timestamp, self.accz)
        self.min_len = min(len(self.p_max), len(self.p_min))
        self.p2_max = [self.p_max[0]]
        self.p2_min = [self.p_min[0]]
        for i in range(1,len(self.p_max)):
            if self.p_max[i]-self.p2_max[-1] >= 4:
                self.p2_max.append(self.p_max[i])
        for i in range(1,len(self.p_min)):
            if self.p_min[i]-self.p2_min[-1] >= 4:
                self.p2_min.append(self.p_min[i])
        self.p_max = self.p2_max
        self.p_min = self.p2_min
        self.min_len = min(len(self.p_max), len(self.p_min))
        a_max = 0
        a_min = 0
        for i in range(self.min_len):
            # a_max += self.acc[self.p_max[i]]
            # a_min += self.acc[self.p_min[i]]
            # a_max /= min_len
            # a_min /= min_len
            delta = np.abs(self.acc[self.p_max[i]] - self.acc[self.p_min[i]]) / 16384 * 9.8
            self.step_length.append(self.K * math.pow(delta, 0.25))
        if self.model == 0:
            for i in self.p_max:
                L.append(self.timestamp[i])
                y.append(self.acc[i])
            self.x2_position = []
            self.y2_position = []
            x_tmp = -1
            y_tmp = 3.4
            # x_tmp = self.x_position[0]
            # y_tmp = self.y_position[0]
            self.x2_position.append(x_tmp)
            self.y2_position.append(y_tmp)
            # if self.model == 0:
            step = 0
            for i in self.p_max:
                if step<self.min_len:

                    nx = -1*np.cos((self.hangxiang[i]+self.hangxiang[i])/2)
                    ny = -1*np.sin((self.hangxiang[i]+self.hangxiang[i])/2)
                    x_tmp = x_tmp + ny*self.step_length[step]
                    y_tmp = y_tmp + nx*self.step_length[step]
                    self.x2_position.append(x_tmp)
                    self.y2_position.append(y_tmp)
                    step +=1
                    self.index.append(i)
                    # self.direction.append(self.fai[i])
                    # self.print_accx.append(self.accx[i])
                    # self.print_accy.append(self.accy[i])
                    # self.print_accz.append(self.accz[i])
                    # self.print_gyrx.append(self.gyroscopex[i])
                    # self.print_gyry.append(self.gyroscopey[i])
                    # self.print_gyrz.append(self.gyroscopez[i])
        else:
            q_max = __find_peak__(self.gyroscopey,5)
            q_min = __find_peak__(-1 * self.gyroscopey,5)

            q_max = np.append(q_max,q_min)
            q_max = sorted(q_max)
            # plt.plot(self.timestamp, self.accz)
            for i in self.p_max:
                L.append(self.timestamp[i])
            a_max = 0
            a_min = 0
            self.x2_position = []
            self.y2_position = []
            x_tmp = -1
            y_tmp = 3.4
            # x_tmp = self.x_position[0]
            # y_tmp = self.y_position[0]
            self.x2_position.append(x_tmp)
            self.y2_position.append(y_tmp)
            # if self.model == 0:
            k = 0
            for i in q_max:
                nx = -1 * np.cos((self.fai[i]+self.fai[i])/2)
                ny = -1 * np.sin((self.fai[i]+self.fai[i])/2)
                x_tmp = x_tmp + ny * self.step_length[k]
                y_tmp = y_tmp + nx * self.step_length[k]
                self.x2_position.append(x_tmp)
                self.y2_position.append(y_tmp)
                self.index.append(i)
                k+=1
                if k>=len(self.step_length):
                    k-=1
        # else:
        #     for i in self.p_max:
        #         nx = -1*np.cos(self.theta[i])
        #         ny = -1*np.sin(self.theta[i])
        #         x_tmp = x_tmp + ny*self.step_length
        #         y_tmp = y_tmp + nx*self.step_length
        #         self.x2_position.append(x_tmp)
        #         self.y2_position.append(y_tmp)

        # print(self.x2_position)
        # print(self.y2_position)
        #因为第一种情况
        # print(len(L))
        # print(L)
        # print(len(q_max))
        # plt.scatter(L, y, color="red")
        # plt.show()
    def plot_acc(self):
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 14,}

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(4, 1, 1)
        # 绘图 # 具体线型、颜色、label可搜索该函数参数
        ax.plot(self.timestamp, self.accx, color='r')
        ax.set_xlabel(r'timestamp', font1)
        ax.set_ylabel("g", font1)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # 添加子图 2，具体方法与图 1 差别不大
        ax2 = fig.add_subplot(4, 1, 2)
        # plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
        ax2.plot(self.timestamp, self.accy, color='red')  # o-:圆形
        labels = ax2.get_xticklabels() + ax2.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        ax3 = fig.add_subplot(4, 1, 3)
        # plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
        ax3.plot(self.timestamp, self.accz, color='blue')  # o-:圆形
        labels = ax2.get_xticklabels() + ax2.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]


        ax3 = fig.add_subplot(4, 1, 4)
        # plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
        ax3.plot(self.timestamp, self.acc, color='blue')  # o-:圆形
        labels = ax2.get_xticklabels() + ax2.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.show()
    def save(self):
        # with open(r'position'str(self.sample_batch)+'.txt', 'a+', encoding='utf-8') as test:
        #     test.truncate(0)

        for i in range(len(self.x2_position) - 1):
            self.length.append(np.sqrt((self.x2_position[i] - self.x2_position[i + 1]) ** 2 + (
                        self.y2_position[i] - self.y2_position[i + 1]) ** 2))
            theta = np.arctan2(self.x2_position[i] - self.x2_position[i + 1],
                               self.y2_position[i] - self.y2_position[i + 1])
            if theta > 2:
                theta = 2 * np.pi - theta
            self.direction.append(theta)
        self.direction.append(theta)
        self.direction = np.array(self.direction)
        self.length.append(np.sqrt((self.x2_position[i] - self.x2_position[i + 1]) ** 2 + (
                self.y2_position[i] - self.y2_position[i + 1]) ** 2))
        # writer.writerow(['x2_position', 'y2_position', 'x_true_position','y_true_position','direction',\
        #                 'print_accx','print_accy','print_accz','print_gyrx','print_gyry','print_gyrz'])

            # data = ([self.x2_position,self.y2_position,self.x_true_position,self.y_true_position,\
            #         self.direction,self.print_accx,self.print_accy,self.print_accz,\
            #         self.print_gyrx,self.print_gyry,self.print_gyrz])

        k = self.index_begin
        self.print_accx.append(self.accx[k])
        self.print_accy.append(self.accy[k])
        self.print_accz.append(self.accz[k])
        self.print_gyrx.append(self.gyroscopex[k])
        self.print_gyry.append(self.gyroscopey[k])
        self.print_gyrz.append(self.gyroscopez[k])
        for i in range(len(self.x2_position)-1):
            k = self.index[i]+self.index_begin
            # self.direction.append(self.fai[i])
            self.print_accx.append(self.accx[k])
            self.print_accy.append(self.accy[k])
            self.print_accz.append(self.accz[k])
            self.print_gyrx.append(self.gyroscopex[k])
            self.print_gyry.append(self.gyroscopey[k])
            self.print_gyrz.append(self.gyroscopez[k])
        txt = ["x", "y", "xtrue", "ytrue","direction", "accx", "accy", "accz", "gyrx", "gyry", "gyrz", "length", "error",
               "sampleBatch"]
        big_data = []
        with open('data_' + str(self.sample_batch) + '.json', 'w', encoding='utf-8') as file:
            aadada = 0
        for i in range(len(self.x2_position)):
            dic = {}
            data = [self.x2_position[i],self.y2_position[i],self.x_2true_position[i],self.y_2true_position[i],\
                self.direction[i],self.print_accx[i],self.print_accy[i],self.print_accz[i],\
                self.print_gyrx[i],self.print_gyry[i],self.print_gyrz[i],self.length[i],self.error[i],self.sample_batch]
            # data = data.encode(encoding='utf_8', errors='strict')
            for j in range(len(txt)):
                dic[txt[j]] = float(data[j])
            big_data.append(dic)
        with open('data_'+str(self.sample_batch)+'.json', 'w', encoding='utf-8') as file:
            json.dump(big_data, file, ensure_ascii=False)
        position = []
        for i in range(len(self.x_position)):
            dic = {}
            data = [self.x_position[i], self.y_position[i]]
            # data = data.encode(encoding='utf_8', errors='strict')
            for j in range(len(data)):
                dic[txt[j]] = float(data[j])
            position.append(dic)
        with open('position_'+str(self.sample_batch)+'.json', 'w', encoding='utf-8') as file:
            json.dump(position, file, ensure_ascii=False)

    def plot_gro(self):
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 14,}
        # 选取画布，figsize控制画布大小
        fig2 = plt.figure(figsize=(7, 5))
        # fig = plt.figure()

        # 绘制子图 1,2,1 代表绘制 1x2 个子图，本图为第 1 个，即 121
        # ax 为本子图
        ax4 = fig2.add_subplot(3, 1, 1)
        # 绘图 # 具体线型、颜色、label可搜索该函数参数
        ax4.plot(self.timestamp, self.gyroscopex, color='r')
        # ax 子图的 x,y 坐标 label 设置
        # 使用 r'...$\gamma$' 的形式可引入数学符号
        # font 为label字体设置
        ax4.set_xlabel(r'timestamp', font1)
        ax4.set_ylabel("g", font1)
        # 坐标轴字体设置
        labels = ax4.get_xticklabels() + ax4.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        # 添加子图 2，具体方法与图 1 差别不大
        ax5 = fig2.add_subplot(3, 1, 2)
        # plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
        ax5.plot(self.timestamp, self.gyroscopey, color='red')  # o-:圆形
        labels = ax5.get_xticklabels() + ax5.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        ax6 = fig2.add_subplot(3, 1, 3)
        # plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
        ax6.plot(self.timestamp, self.gyroscopez, color='blue')  # o-:圆形
        labels = ax6.get_xticklabels() + ax6.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # 保存及展示
        # plt.savefig('hyperparams.eps')
        plt.show()
    def plot_angle(self):
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 14,}
        # 选取画布，figsize控制画布大小
        fig2 = plt.figure(figsize=(7, 5))
        # fig = plt.figure()

        # 绘制子图 1,2,1 代表绘制 1x2 个子图，本图为第 1 个，即 121
        # ax 为本子图
        ax4 = fig2.add_subplot(4, 1, 1)
        # 绘图 # 具体线型、颜色、label可搜索该函数参数
        ax4.plot(self.timestamp, self.theta, color='r')
        # ax 子图的 x,y 坐标 label 设置
        # 使用 r'...$\gamma$' 的形式可引入数学符号
        # font 为label字体设置
        ax4.set_xlabel(r'timestamp', font1)
        ax4.set_ylabel("g", font1)
        # 坐标轴字体设置
        labels = ax4.get_xticklabels() + ax4.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        # 添加子图 2，具体方法与图 1 差别不大
        ax5 = fig2.add_subplot(4, 1, 2)
        # plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
        ax5.plot(self.timestamp, self.gama, color='red')  # o-:圆形
        labels = ax5.get_xticklabels() + ax5.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        ax6 = fig2.add_subplot(4, 1, 3)
        # plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
        ax6.plot(self.timestamp, self.fai, color='blue')  # o-:圆形
        labels = ax6.get_xticklabels() + ax6.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        ax6 = fig2.add_subplot(4, 1, 4)
        # plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
        ax6.plot(self.timestamp, self.hangxiang, color='blue')  # o-:圆形
        labels = ax6.get_xticklabels() + ax6.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # 保存及展示
        # plt.savefig('hyperparams.eps')
        plt.show()
    def z_inter(self):
        jifen = []
        tmp = 0
        for i in range(len(self.timestamp)):
            tmp = tmp +self.gyroscopez[i]*self.timestamp[i]
            jifen.append(tmp/180*3.1415926)
        plt.scatter(self.timestamp, jifen, color="red")
        # print(jifen)
        plt.show()

    def error(self):
        self.error = []
        for i in range(len(self.x2_position)):
            self.error.append(np.sqrt((self.x2_position[i]-self.x_2true_position[i])**2+(self.y2_position[i]-self.y_2true_position[i])**2))
        error_avg = np.average(self.error)
        # print(str(self.sample_batch)+"    "+ str(error_avg))
        lenaa = len(self.error)
        sorted_data = np.sort(self.error)
        error_50 = sorted_data[int(0.5 * lenaa)]
        error_75 = sorted_data[int(0.75 * lenaa)]
        error_90 = sorted_data[int(0.9 * lenaa)]
        aa = [int(self.sample_batch),error_avg,error_50,error_75,error_90]
        # (error_50,error_75,error_90) = CDF(self.error,self.sample_batch)
        with open('error_'+str(self.sample_batch)+'.json', 'w', encoding='utf-8') as file:
            dic = {}
            txt = ["sampleBatch","average_error", "error_50", "error_75","error_90"]
            for i in range(len(txt)):
                dic[txt[i]] = float(aa[i])
            d = [dic]
            json.dump(d, file, ensure_ascii=False)
            file.write('\n')
    def get_truth(self,truth_position):
        self.x2_position = np.array(self.x2_position)
        self.y2_position = np.array(self.y2_position)
        with open(truth_position) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            # header = next(csv_reader)        # 读取第一行每一列的标题
            begin = 0
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                if begin == 0:
                    begin = 1
                    continue
                # if row[0] in ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']:
                self.x_true_position.append(eval(row[1]))
                self.y_true_position.append(eval(row[2]))
        self.x_2true_position.append(self.x_true_position[0])
        self.y_2true_position.append(self.y_true_position[0])
        for i in range(1, len(self.x_true_position)):
            self.x_2true_position.append((self.x_true_position[i - 1] + self.x_true_position[i]) / 2)
            self.x_2true_position.append(self.x_true_position[i])
            self.y_2true_position.append((self.y_true_position[i - 1] + self.y_true_position[i]) / 2)
            self.y_2true_position.append(self.y_true_position[i])
        delta = len(self.x2_position) - len(self.x_2true_position)
        # 去掉多的点
        if delta > 0:
            for i in range(0,delta+1):
                error = 0
                x_pos_bias = self.x_2true_position[0] - self.x2_position[i]
                y_pos_bias = self.y_2true_position[0] - self.y2_position[i]
                self.x2_position += x_pos_bias
                self.y2_position += y_pos_bias
                for k in range(len(self.x_2true_position)):
                    error+= np.sqrt((self.x_2true_position[k]-self.x2_position[k+i])**2+(self.y_2true_position[k]-self.y2_position[k+i])**2)
                self.match_error.append(error)
            min_index = 0
            min = 100
            for j in range(delta+1):
                if self.match_error[j]<min:
                    min = self.match_error[j]
                    min_index = j
            x_pos_bias = self.x_2true_position[0] - self.x2_position[min_index]
            y_pos_bias = self.y_2true_position[0] - self.y2_position[min_index]
            self.x2_position += x_pos_bias
            self.y2_position += y_pos_bias
            self.x2_position = self.x2_position[min_index:min_index + len(self.x_2true_position)]
            self.y2_position = self.y2_position[min_index:min_index + len(self.y_2true_position)]
            # len(self.x_position)
            if min_index == 0 :
                self.index = self.index[min_index:min_index+len(self.x_2true_position)-1]
            else:
                self.index_begin = self.index[min_index-1]
                self.index = self.index[min_index:min_index+len(self.x_2true_position)-1]
                self.index = self.index - self.index_begin
            # index 少保存一个，因为起始点对应的不需要，一个index对应一个波峰，对应一步，所以比点数少1

        #先把如果多于29减下去
        index_length = self.index[-1]
        index_delta = index_length/(len(self.y_true_position)-1)
        self.x_2true_position = []
        self.y_2true_position = []



        # print(self.index)
        # print(len(self.index))
        # print(index_length)
        # print(index_delta)
        self.x_2true_position.append(self.x_true_position[0])
        self.y_2true_position.append(self.y_true_position[0])
        for i in range(0,len(self.x2_position)-2):
            left = int((self.index[i]-0)/index_delta)
            k = (self.index[i])/index_delta-left
            # print(i,left)
            self.x_2true_position.append(((1-k)*self.x_true_position[left]+k*self.x_true_position[left+1]))
            self.y_2true_position.append(((1-k)*self.y_true_position[left]+k*self.y_true_position[left+1]))
        self.x_2true_position.append(self.x_true_position[-1])
        self.y_2true_position.append(self.y_true_position[-1])
        # data_step_numbers = len(self.x2_position)
        # true_step_numbers = len(self.x_2true_position)
        # delta = data_step_numbers-true_step_numbers
        # self.match_error = []

        min_index = 0
        min = 500
        # self.begin_error = []
        for i in range(0, int(len(self.x_position)/2)):
            error = 0
            x_pos_bias = self.x_position[i] - self.x2_position[0]
            y_pos_bias = self.y_position[i] - self.y2_position[0]
            self.x2_position += x_pos_bias
            self.y2_position += y_pos_bias
            for k in range(len(self.x_2true_position)):
                error +=  np.sqrt((self.x_2true_position[k] - self.x2_position[k]) ** 2 + (self.y_2true_position[k] - self.y2_position[k]) ** 2)
            if error < min:
                min_index = i
                min = error
            # self.begin_error.append(error)
            # self.match_error[j] < min:
        # print(min)
        x_pos_bias = self.x_position[min_index] - self.x2_position[0]
        y_pos_bias = self.y_position[min_index] - self.y2_position[0]
        self.x2_position += x_pos_bias
        self.y2_position += y_pos_bias
            # print(len(self.x_position))
            # error = 0
            # for k in range(true_step_numbers):
            #     error += np.sqrt((self.x_2true_position[k] - self.x2_position[k]) ** 2 + (self.y_2true_position[k] - self.y2_position[k]) ** 2)
            # i = 1
            # self.x3_position = []
            # self.y3_position = []
            # self.index2 = [0]
            # self.x3_position.append(self.x2_position[0])
            # self.y3_position.append(self.y2_position[0])
            # while 2*i<len(self.x2_position):
            #     self.index2.append(self.index[2*i-1])
            #     self.x3_position.append(self.x2_position[i*2])
            #     self.y3_position.append(self.y2_position[i*2])
            #     i += 1
            # self.x2_position =  self.x3_position
            # self.y2_position = self.y3_position
            # self.index = self.index2


def Vsqrt(L):
    L = np.array(L)
    theta_speed = np.sqrt(np.sum(L ** 2))
    return L/theta_speed

def batch_time(batch,csv_postion,csv_running,truth_position,model):
    sample_batch = batch
    P1 = PDR(csv_postion,csv_running,sample_batch,model)
    # P1.plot_acc()
    # P1.plot_gro()
    # P1.Coordinate_system_rotation()
    P1.vis2()
    P1.Coordinate_system_rotation()
    # P1.plot_gro()
    # P1.plot_angle()
    # P1.z_inter()
    P1.get_truth(truth_position)
    P1.error()
    # print(P1.step_length)
    print(np.average(P1.error))
    P1.save()
    # # P1.z_jifen()
    # img = plt.imread('C:/Users/zzzzzzzzz/Desktop/学习/软件课设/软件课程设计2022资料/室内地图/1.jpg',5)
    # fig,ax = plt.subplots(figsize=(7,7),dpi=200)
    # ax.imshow(img,extent=[-7.1+1.65,7.1+1.65,-5.2,6])
    # # print(P1.y_position)
    # ax.plot(P1.x_position,P1.y_position,color='b')
    img = plt.imread('D:/code/PycharmProjects/software/1.jpg',5)
    fig,ax = plt.subplots(figsize=(7,7),dpi=200)
    ax.imshow(img,extent=[-7.1+1.65,7.1+1.65,-5.2,6])
    ax.plot(P1.x2_position,P1.y2_position,color='r')
    ax.plot(P1.x_true_position,P1.y_true_position,color='y')
#下面是测试的两个数据集。
# csv_postion = 'D:/code/PycharmProjects/software/平持/position.csv'
# csv_running = 'D:/code/PycharmProjects/software/平持/running.csv'
# truth_position = 'D:/code/PycharmProjects/software/平持/ground_truth.csv'
#
#
# batch_time('90',csv_postion,csv_running,truth_position,'0')
# csv_postion = 'D:/code/PycharmProjects/software/摇摆/position.csv'
# csv_running = 'D:/code/PycharmProjects/software/摇摆/running.csv'
# truth_position = 'D:/code/PycharmProjects/software/摇摆/ground_truth.csv'
# batch_time('79',csv_postion,csv_running,truth_position,'1')
csv_postion = 'D:/code/PycharmProjects/software/position.csv'
csv_running = 'D:/code/PycharmProjects/software/running.csv'
truth_position = 'D:/code/PycharmProjects/software/ground_truth.csv'
batch_time('27',csv_postion,csv_running,truth_position,'0')
batch_time('28',csv_postion,csv_running,truth_position,'0')
batch_time('29',csv_postion,csv_running,truth_position,'0')
batch_time('30',csv_postion,csv_running,truth_position,'1')
batch_time('31',csv_postion,csv_running,truth_position,'1')
batch_time('32',csv_postion,csv_running,truth_position,'1')
# 选batch 和模式 0代表平持，1代表摇摆，三个csv路径
plt.show()

