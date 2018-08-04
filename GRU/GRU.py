# coding=utf-8
# coding=utf-8
import os
import getTraindata
import math
import random
import function
import ecs
# 初始化网络结构
def GRUtrain(traindata1,zhenshidata1,k):
    # 初始化网络结构
    uNum = k  # 数据结构单元个数
    xdim = 1  #输入数据维度是1
    ydim = 1  #输出维度也是1
    hdim = k
    eta = 0.1  # 学习率

    #训练数据

    traindata = [[0 for col in range(len(traindata1[0]))] for row in range(len(traindata1))]
    for i in range(len(traindata1)):
        for j in range(len(traindata1[0])):
            if(max(traindata1[i])==min(traindata1[i]) and max(traindata1[i])==0):
                traindata[i][j]=0.0
            if(max(traindata1[i])==min(traindata1[i]) and max(traindata1[i])!=0):
                traindata[i][j]=1.0
            if (max(traindata1[i]) != min(traindata1[i])):
                traindata[i][j] = (traindata1[i][j]-min(traindata1[i]))/float(max(traindata1[i])-min(traindata1[i]))
    #print(traindata)
    zhenshidata = [[0 for col in range(len(zhenshidata1[0]))] for row in range(len(zhenshidata1))]
    for i in range(len(zhenshidata1)):
        for j in range(len(zhenshidata1[0])):
            if (max(zhenshidata1[i]) == min(zhenshidata1[i]) and max(zhenshidata1[i]) == 0):
                zhenshidata[i][j] = 0.0
            if (max(zhenshidata1[i]) == min(zhenshidata1[i]) and max(zhenshidata1[i]) != 0):
                zhenshidata[i][j] = 1.0
            if (max(zhenshidata1[i]) != min(zhenshidata1[i])):
                zhenshidata[i][j] = (zhenshidata1[i][j]-min(zhenshidata1[i]))/float(max(zhenshidata1[i])-min(zhenshidata1[i]))


    # 初始化网络参数，开始训练
    Wy = [0 for i in range(hdim)]
    for i in range(hdim):
        Wy[i] = random.uniform(-1, 1)

    Wr = [0 for i in range(hdim)]
    for i in range(hdim):
        Wr[i] = random.uniform(-1, 1)
    Ur = [[0 for col in range(hdim)] for row in range(hdim)]
    for i in range(hdim):
        for j in range(hdim):
            Ur[i][j] = random.uniform(-1, 1)

    W = [0 for i in range(hdim)]
    for i in range(hdim):
        W[i] = random.uniform(-1, 1)
    U = [[0 for col in range(hdim)] for row in range(hdim)]
    for i in range(hdim):
        for j in range(hdim):
            U[i][j] = random.uniform(-1, 1)

    Wz = [0 for i in range(hdim)]
    for i in range(hdim):
        Wz[i] = random.uniform(-1, 1)
    Uz = [[0 for col in range(hdim)] for row in range(hdim)]
    for i in range(hdim):
        for j in range(hdim):
            Uz[i][j] = random.uniform(-1, 1)
    # cell数据存储变量
    rvalues = [[0 for col in range(hdim)] for row in range(uNum + 1)]
    zvalues = [[0 for col in range(hdim)] for row in range(uNum + 1)]
    hbarvalues = [[0 for col in range(hdim)] for row in range(uNum)]
    hvalues = [[0 for col in range(hdim)] for row in range(uNum)]
    yvalues = [0 for i in range(uNum)]
    yyvalues=[]
    yxvalues=[]
    pianchaall = []

    for i in range(len(traindata)):
        # 前向计算
        rvalues[0] = function.sigmoid(function.xchenlist(traindata[i][0], Wr))
        hbarvalues[0] = function.tanh(function.xchenlist(traindata[i][0], W))
        zvalues[0] = function.sigmoid(function.xchenlist(traindata[i][0], Wz))
        hvalues[0] = function.listchenlist(zvalues[0], hbarvalues[0])
        yvalues[0] = function.sigmoid(function.hangchenlie(hvalues[0], Wy))
        for t in range(1, uNum):
            rvalues[t] = function.sigmoid(function.listjialist(function.xchenlist(traindata[i][t], Wr),
                                                               function.hangchenjuzhen(hvalues[t - 1], Ur)))
            hbarvalues[t] = function.tanh(function.listjialist(function.xchenlist(traindata[i][t], W),
                                                               function.hangchenjuzhen(
                                                                   function.listchenlist(rvalues[t], hvalues[t - 1]),
                                                                   U)))
            zvalues[t] = function.sigmoid(function.listjialist(function.xchenlist(traindata[i][t], Wz),
                                                               function.hangchenjuzhen(hvalues[t - 1], Uz)))
            hvalues[t] = function.listjialist(function.listchenlist(function.kjianlist(1, zvalues[t]), hvalues[t - 1]),
                                              function.listchenlist(zvalues[t], hbarvalues[t]))
            yvalues[t] = function.sigmoid(function.hangchenlie(hvalues[t], Wy))
        # 反向传播
        delta_r_next = [0 for i in range(hdim)]
        delta_z_next = [0 for i in range(hdim)]
        delta_h_next = [0 for i in range(hdim)]
        delta_next = [0 for i in range(hdim)]

        dWy = [0 for i in range(hdim)]
        dWr = [0 for i in range(hdim)]
        dUr = [[0 for col in range(hdim)] for row in range(hdim)]
        dW = [0 for i in range(hdim)]
        dU = [[0 for col in range(hdim)] for row in range(hdim)]
        dWz = [0 for i in range(hdim)]
        dUz = [[0 for col in range(hdim)] for row in range(hdim)]

        for t in range(1, uNum):
            delta_y = yvalues[uNum - t] - float(zhenshidata[i][uNum - t]) * function.sigmoidtoD(yvalues[uNum - t])
            #print(delta_y)
            a = function.xchenlist(delta_y, Wy)
            b = function.hangchenjuzhen(delta_z_next, zip(*Uz))
            c = function.listchenlist(function.hangchenjuzhen(delta_next, zip(*U)), rvalues[uNum - t + 1])
            d = function.hangchenjuzhen(delta_r_next, zip(*Ur))
            e = function.listchenlist(delta_h_next, function.kjianlist(1, zvalues[uNum - t + 1]))
            delta_h = function.listjialist(function.listjialist(function.listjialist(function.listjialist(a, b), c), d),
                                           e)
            # print(delta_h)
            delta_z = function.listchenlist(
                function.listchenlist(delta_h, function.listjianlist(hbarvalues[uNum - t], hvalues[uNum - t - 1])),
                function.sigmoidtoD(zvalues[uNum - t]))
            # print(delta_z)
            delta = function.listchenlist(function.listchenlist(delta_h, zvalues[uNum - t]),
                                          function.tanhtoD(hbarvalues[uNum - t]))
            # print(delta)
            delta_r = function.listchenlist(function.hangchenjuzhen(function.listchenlist(
                function.listchenlist(hvalues[uNum - t - 1], function.listchenlist(delta_h, zvalues[uNum - t])),
                function.tanhtoD(hbarvalues[uNum - t])), zip(*U)), function.sigmoidtoD(rvalues[uNum - t]))
            # print(delta_r)

            dWy = dWy + function.xchenlist(delta_y, hvalues[uNum - t])
            dWz = dWz + function.xchenlist(traindata[i][uNum - t], delta_z)
            dUz = dUz + function.liechenhang(hvalues[uNum - t - 1], delta_z)
            dW = dW + function.xchenlist(traindata[i][uNum - t], delta)
            dU = dU + function.liechenhang(function.listchenlist(rvalues[uNum - t], hvalues[uNum - t - 1]), delta)
            dWr = dWr + function.xchenlist(traindata[i][uNum - t], delta_r)
            dUr = dUr + function.liechenhang(hvalues[uNum - t - 1], delta_r)
            delta_r_next = delta_r
            delta_z_next = delta_z
            delta_h_next = delta_h
            delta_next = delta
        t = uNum
        delta_y = yvalues[uNum - t] - float(zhenshidata[i][uNum - t]) * function.sigmoidtoD(yvalues[uNum - t])
        # print(delta_y)
        a = function.xchenlist(delta_y, Wy)
        b = function.hangchenjuzhen(delta_z_next, zip(*Uz))
        c = function.listchenlist(function.hangchenjuzhen(delta_next, zip(*U)), rvalues[uNum - t + 1])
        d = function.hangchenjuzhen(delta_r_next, zip(*Ur))
        e = function.listchenlist(delta_h_next, function.kjianlist(1, zvalues[uNum - t + 1]))
        delta_h = function.listjialist(function.listjialist(function.listjialist(function.listjialist(a, b), c), d), e)
        # print(delta_h)
        delta_z = function.listchenlist(function.listchenlist(delta_h, hbarvalues[uNum - t]),
                                        function.sigmoidtoD(zvalues[uNum - t]))
        delta = function.listchenlist(function.listchenlist(delta_h, zvalues[uNum - t]),
                                      function.tanhtoD(hbarvalues[uNum - t]))
        delta_r = [0 for k in range(hdim)]
        # print(delta)

        dWy = dWy + function.xchenlist(delta_y, hvalues[uNum - t])
        dWz = dWz + function.xchenlist(traindata[i][uNum - t], delta_z)

        dW = dW + function.xchenlist(traindata[i][uNum - t], delta)

        dWr = dWr + function.xchenlist(traindata[i][uNum - t], delta_r)
        Wy = function.listjianlist(Wy, function.xchenlist(eta, dWy))
        Wr = function.listjianlist(Wr, function.xchenlist(eta, dWr))
        W = function.listjianlist(W, function.xchenlist(eta, dW))
        Wz = function.listjianlist(Wz, function.xchenlist(eta, dWz))

        Ur = function.juzhenjianjuzhen(Ur, function.xchenjuzhen(eta, dUr))
        U = function.juzhenjianjuzhen(U, function.xchenjuzhen(eta, dU))
        Uz = function.juzhenjianjuzhen(Uz, function.xchenjuzhen(eta, dUz))
        x=function.kjialist(min(traindata1[i]), function.xchenlist((max(traindata1[i]) - min(traindata1[i])), yvalues))
        piancha1 = function.listjianlist(zhenshidata[i], yvalues)
        piancha2=function.listjianlist(zhenshidata1[i], x)
        pianchaall.append(piancha2)



        error1 = (math.sqrt(sum(function.listchenlist(piancha1, piancha1)))) / 2.0
        error2 = (math.sqrt(sum(function.listchenlist(piancha2, piancha2)))) / 2.0

        # 打印真实shuju
        #print(zhenshidata[i])
        # 打印预测数据
        #print(yvalues)
        yyvalues.append(function.kjialist(min(traindata1[i]),function.xchenlist((max(traindata1[i])-min(traindata1[i])),yvalues)))
        yxvalues.append(yvalues)
        #print(yyvalues[i])
        # 打印误差
        #print(error1)
        #print(piancha2)
        #print(error2)
    #print(pianchaall)
    #print(yyvalues)
    w=[0 for i in range(hdim)]
    pianchabu=[]
    ss=zip(*pianchaall)
    #print(ss)
    for jj in range(hdim):
        pianchabu.append(sum(ss[jj])/float(len(traindata1)))
    #print(pianchabu)
    for i in range(hdim):
        w[i]=random.uniform(-1, 1)
    #print(yxvalues)
    #print(zhenshidata)
    #print(zhenshidata[0][hdim-1])
    for j in range(hdim):
        x=0
        for kk in range(len(yyvalues)):
            x=x+(zhenshidata[kk][hdim-1]-function.hangchenlie(w,yxvalues[kk]))*yxvalues[kk][j]
        w[j]=w[j]+eta*x
    #print(w)
    return Wy,W,U,Wz,Uz,Wr,Ur,w,pianchabu

def GRUtrain2(traindata1,zhenshidata1,k,Wy,W,U,Wz,Uz,Wr,Ur,w):
    # 初始化网络结构
    uNum = k  # 数据结构单元个数
    xdim = 1  #输入数据维度是1
    ydim = 1  #输出维度也是1
    hdim = k
    eta = 0.1  # 学习率

    #训练数据
    traindata = [[0 for col in range(len(traindata1[0]))] for row in range(len(traindata1))]
    for i in range(len(traindata1)):
        for j in range(len(traindata1[0])):
            if (max(traindata1[i]) == min(traindata1[i]) and max(traindata1[i]) == 0):
                traindata[i][j] = 0.0
            if (max(traindata1[i]) == min(traindata1[i]) and max(traindata1[i]) != 0):
                traindata[i][j] = 1.0
            if (max(traindata1[i]) != min(traindata1[i])):
                traindata[i][j] = (traindata1[i][j] - min(traindata1[i])) / float(
                    max(traindata1[i]) - min(traindata1[i]))
    # print(traindata)
    zhenshidata = [[0 for col in range(len(zhenshidata1[0]))] for row in range(len(zhenshidata1))]
    for i in range(len(zhenshidata1)):
        for j in range(len(zhenshidata1[0])):
            if (max(zhenshidata1[i]) == min(zhenshidata1[i]) and max(zhenshidata1[i]) == 0):
                zhenshidata[i][j] = 0.0
            if (max(zhenshidata1[i]) == min(zhenshidata1[i]) and max(zhenshidata1[i]) != 0):
                zhenshidata[i][j] = 1.0
            if (max(zhenshidata1[i]) != min(zhenshidata1[i])):
                zhenshidata[i][j] = (zhenshidata1[i][j] - min(zhenshidata1[i])) / float(
                    max(zhenshidata1[i]) - min(zhenshidata1[i]))


    # cell数据存储变量
    rvalues = [[0 for col in range(hdim)] for row in range(uNum + 1)]
    zvalues = [[0 for col in range(hdim)] for row in range(uNum + 1)]
    hbarvalues = [[0 for col in range(hdim)] for row in range(uNum)]
    hvalues = [[0 for col in range(hdim)] for row in range(uNum)]
    yvalues = [0 for i in range(uNum)]
    #print(yvalues)
    yyvalues=[]
    yxvalues=[]
    pianchaall = []

    for i in range(len(traindata)):
        # 前向计算
        rvalues[0] = function.sigmoid(function.xchenlist(traindata[i][0], Wr))
        hbarvalues[0] = function.tanh(function.xchenlist(traindata[i][0], W))
        zvalues[0] = function.sigmoid(function.xchenlist(traindata[i][0], Wz))
        hvalues[0] = function.listchenlist(zvalues[0], hbarvalues[0])
        yvalues[0] = function.sigmoid(function.hangchenlie(hvalues[0], Wy))
        for t in range(1, uNum):
            rvalues[t] = function.sigmoid(function.listjialist(function.xchenlist(traindata[i][t], Wr),
                                                               function.hangchenjuzhen(hvalues[t - 1], Ur)))
            hbarvalues[t] = function.tanh(function.listjialist(function.xchenlist(traindata[i][t], W),
                                                               function.hangchenjuzhen(
                                                                   function.listchenlist(rvalues[t], hvalues[t - 1]),
                                                                   U)))
            zvalues[t] = function.sigmoid(function.listjialist(function.xchenlist(traindata[i][t], Wz),
                                                               function.hangchenjuzhen(hvalues[t - 1], Uz)))
            hvalues[t] = function.listjialist(function.listchenlist(function.kjianlist(1, zvalues[t]), hvalues[t - 1]),
                                              function.listchenlist(zvalues[t], hbarvalues[t]))
            yvalues[t] = function.sigmoid(function.hangchenlie(hvalues[t], Wy))
        #print(yvalues)
        # 反向传播
        delta_r_next = [0 for i in range(hdim)]
        delta_z_next = [0 for i in range(hdim)]
        delta_h_next = [0 for i in range(hdim)]
        delta_next = [0 for i in range(hdim)]

        dWy = [0 for i in range(hdim)]
        dWr = [0 for i in range(hdim)]
        dUr = [[0 for col in range(hdim)] for row in range(hdim)]
        dW = [0 for i in range(hdim)]
        dU = [[0 for col in range(hdim)] for row in range(hdim)]
        dWz = [0 for i in range(hdim)]
        dUz = [[0 for col in range(hdim)] for row in range(hdim)]
        #print(yvalues)
        #print(uNum)

        for t in range(1, uNum):
            delta_y = yvalues[uNum - t] - float(zhenshidata[i][uNum - t]) * function.sigmoidtoD(yvalues[uNum - t])
            # print(delta_y)
            a = function.xchenlist(delta_y, Wy)
            b = function.hangchenjuzhen(delta_z_next, zip(*Uz))
            c = function.listchenlist(function.hangchenjuzhen(delta_next, zip(*U)), rvalues[uNum - t + 1])
            d = function.hangchenjuzhen(delta_r_next, zip(*Ur))
            e = function.listchenlist(delta_h_next, function.kjianlist(1, zvalues[uNum - t + 1]))
            delta_h = function.listjialist(function.listjialist(function.listjialist(function.listjialist(a, b), c), d),
                                           e)
            # print(delta_h)
            delta_z = function.listchenlist(
                function.listchenlist(delta_h, function.listjianlist(hbarvalues[uNum - t], hvalues[uNum - t - 1])),
                function.sigmoidtoD(zvalues[uNum - t]))
            # print(delta_z)
            delta = function.listchenlist(function.listchenlist(delta_h, zvalues[uNum - t]),
                                          function.tanhtoD(hbarvalues[uNum - t]))
            # print(delta)
            delta_r = function.listchenlist(function.hangchenjuzhen(function.listchenlist(
                function.listchenlist(hvalues[uNum - t - 1], function.listchenlist(delta_h, zvalues[uNum - t])),
                function.tanhtoD(hbarvalues[uNum - t])), zip(*U)), function.sigmoidtoD(rvalues[uNum - t]))
            # print(delta_r)

            dWy = dWy + function.xchenlist(delta_y, hvalues[uNum - t])
            dWz = dWz + function.xchenlist(traindata[i][uNum - t], delta_z)
            dUz = dUz + function.liechenhang(hvalues[uNum - t - 1], delta_z)
            dW = dW + function.xchenlist(traindata[i][uNum - t], delta)
            dU = dU + function.liechenhang(function.listchenlist(rvalues[uNum - t], hvalues[uNum - t - 1]), delta)
            dWr = dWr + function.xchenlist(traindata[i][uNum - t], delta_r)
            dUr = dUr + function.liechenhang(hvalues[uNum - t - 1], delta_r)
            delta_r_next = delta_r
            delta_z_next = delta_z
            delta_h_next = delta_h
            delta_next = delta
        t = uNum
        delta_y = yvalues[uNum - t] - float(zhenshidata[i][uNum - t]) * function.sigmoidtoD(yvalues[uNum - t])
        # print(delta_y)
        a = function.xchenlist(delta_y, Wy)
        b = function.hangchenjuzhen(delta_z_next, zip(*Uz))
        c = function.listchenlist(function.hangchenjuzhen(delta_next, zip(*U)), rvalues[uNum - t + 1])
        d = function.hangchenjuzhen(delta_r_next, zip(*Ur))
        e = function.listchenlist(delta_h_next, function.kjianlist(1, zvalues[uNum - t + 1]))
        delta_h = function.listjialist(function.listjialist(function.listjialist(function.listjialist(a, b), c), d), e)
        # print(delta_h)
        delta_z = function.listchenlist(function.listchenlist(delta_h, hbarvalues[uNum - t]),
                                        function.sigmoidtoD(zvalues[uNum - t]))
        delta = function.listchenlist(function.listchenlist(delta_h, zvalues[uNum - t]),
                                      function.tanhtoD(hbarvalues[uNum - t]))
        delta_r = [0 for k in range(hdim)]
        # print(delta)

        dWy = dWy + function.xchenlist(delta_y, hvalues[uNum - t])
        dWz = dWz + function.xchenlist(traindata[i][uNum - t], delta_z)

        dW = dW + function.xchenlist(traindata[i][uNum - t], delta)

        dWr = dWr + function.xchenlist(traindata[i][uNum - t], delta_r)
        Wy = function.listjianlist(Wy, function.xchenlist(eta, dWy))
        Wr = function.listjianlist(Wr, function.xchenlist(eta, dWr))
        W = function.listjianlist(W, function.xchenlist(eta, dW))
        Wz = function.listjianlist(Wz, function.xchenlist(eta, dWz))

        Ur = function.juzhenjianjuzhen(Ur, function.xchenjuzhen(eta, dUr))
        U = function.juzhenjianjuzhen(U, function.xchenjuzhen(eta, dU))
        Uz = function.juzhenjianjuzhen(Uz, function.xchenjuzhen(eta, dUz))
        x=function.kjialist(min(traindata1[i]), function.xchenlist((max(traindata1[i]) - min(traindata1[i])), yvalues))
        piancha1 = function.listjianlist(zhenshidata[i], yvalues)
        piancha2=function.listjianlist(zhenshidata1[i], x)
        pianchaall.append(piancha2)



        error1 = (math.sqrt(sum(function.listchenlist(piancha1, piancha1)))) / 2.0
        error2 = (math.sqrt(sum(function.listchenlist(piancha2, piancha2)))) / 2.0

        # 打印真实shuju
        #print(zhenshidata[i])
        # 打印预测数据
        #print(yvalues)
        yyvalues.append(function.kjialist(min(traindata1[i]),function.xchenlist((max(traindata1[i])-min(traindata1[i])),yvalues)))
        yxvalues.append(yvalues)
        #print(yyvalues[i])
        # 打印误差
        #print(error1)
        #print(piancha2)
        #print(error2)
    #print(pianchaall)
    #print(yyvalues)

    pianchabu=[]
    ss=zip(*pianchaall)
    #print(ss)
    for jj in range(hdim):
        pianchabu.append(sum(ss[jj])/float(len(traindata1)))
    #print(pianchabu)
    #print(yxvalues)
    #print(zhenshidata)
    #print(zhenshidata[0][hdim-1])
    for j in range(hdim):
        x=0
        for kk in range(len(yyvalues)):
            x=x+(zhenshidata[kk][hdim-1]-function.hangchenlie(w,yxvalues[kk]))*yxvalues[kk][j]
        w[j]=w[j]+eta*x
    #print(w)
    return Wy,W,U,Wz,Uz,Wr,Ur,w,pianchabu,error2
def GRUyuce(traindata1,k,Wy,W,U,Wz,Uz,Wr,Ur,w,pianchabu):
    # 初始化网络结构
    uNum = k  # 数据结构单元个数
    hdim = k
    eta = 0.1  # 学习率

    #训练数据

    traindata =[0 for i in range(len(traindata1))]
    for i in range(len(traindata1)):
            if (max(traindata1) == min(traindata1) and max(traindata1) == 0):
                traindata[i] = 0.0
            if (max(traindata1) == min(traindata1) and max(traindata1) != 0):
                traindata[i] = 1.0
            if (max(traindata1) != min(traindata1)):
                traindata[i] = (traindata1[i]-min(traindata1))/float(max(traindata1)-min(traindata1))
    #print(traindata)




    # cell数据存储变量
    rvalues = [[0 for col in range(hdim)] for row in range(uNum + 1)]
    zvalues = [[0 for col in range(hdim)] for row in range(uNum + 1)]
    hbarvalues = [[0 for col in range(hdim)] for row in range(uNum)]
    hvalues = [[0 for col in range(hdim)] for row in range(uNum)]
    yvalues = [0 for i in range(uNum)]



    # 前向计算
    rvalues[0] = function.sigmoid(function.xchenlist(traindata[0], Wr))
    hbarvalues[0] = function.tanh(function.xchenlist(traindata[0], W))
    zvalues[0] = function.sigmoid(function.xchenlist(traindata[0], Wz))
    hvalues[0] = function.listchenlist(zvalues[0], hbarvalues[0])
    yvalues[0] = function.sigmoid(function.hangchenlie(hvalues[0], Wy))
    for t in range(1, uNum):
        rvalues[t] = function.sigmoid(function.listjialist(function.xchenlist(traindata[t], Wr),
                                                               function.hangchenjuzhen(hvalues[t - 1], Ur)))
        hbarvalues[t] = function.tanh(function.listjialist(function.xchenlist(traindata[t], W),
                                                               function.hangchenjuzhen(
                                                                   function.listchenlist(rvalues[t], hvalues[t - 1]),
                                                                   U)))
        zvalues[t] = function.sigmoid(function.listjialist(function.xchenlist(traindata[t], Wz),
                                                               function.hangchenjuzhen(hvalues[t - 1], Uz)))
        hvalues[t] = function.listjialist(function.listchenlist(function.kjianlist(1, zvalues[t]), hvalues[t - 1]),
                                              function.listchenlist(zvalues[t], hbarvalues[t]))
        yvalues[t] = function.sigmoid(function.hangchenlie(hvalues[t], Wy))

        x=function.kjialist(min(traindata1), function.xchenlist((max(traindata1) - min(traindata1)), yvalues))
    yucezhi=function.hangchenlie(w, function.listjialist(x, pianchabu))



    return yucezhi





if __name__ == "__main__":

    historyfile = 'data_2015_12.txt'
    historylins =ecs.read_lines(historyfile)
    Num1 = getTraindata.getTraindata(historylins)
    types=0
    list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    #print(len(list))
    #print(Num1)
    kk=0
    wucha=[0 for i in range(12)]
    #print(Num1)
    for danyuanshu in range(3,5):
       print(danyuanshu)
       listtraindata, listzhenshidata=function.gettraindata(Num1[types],danyuanshu)
       print(listtraindata)
       print(listzhenshidata)
       (Wy, W, U, Wz, Uz, Wr, Ur, w, pianchabu) = GRUtrain(listtraindata, listzhenshidata, danyuanshu)
       print(1111)
       #print(0)
       listtraindata1, listzhenshidata1 = function.gettraindata(list, danyuanshu)
       #print(2)
       print(listtraindata1)
       print(listzhenshidata1)

       (Wy1, W1, U1, Wz1, Uz1, Wr1, Ur1, w1, pianchabu1,error)=GRUtrain2(listtraindata1,listzhenshidata1,danyuanshu,Wy,W,U,Wz,Uz,Wr,Ur,w)
       print(22222)
       #wucha[danyuanshu-3]=error
    for i in range(len(wucha)):
        if(wucha[i]==min(wucha)):
            hdim=i+3
            break


'''
traindata1 = [[10, 12, 11, 11, 0, 16, 12], [10, 12, 11, 11, 0, 16, 13], [10, 12, 11, 11, 0, 16, 12],
                  [10, 12, 11, 11, 0, 16, 12], [10, 12, 11, 11, 0, 16, 12], [10, 12, 11, 11, 0, 16, 12],
                  [10, 12, 11, 11, 0, 16, 12], [10, 12, 11, 11, 0, 16, 12]]
    zhenshidata1 = [[12, 11, 11, 10, 16, 12, 18], [12, 11, 11, 10, 16, 12, 19], [12, 11, 11, 10, 16, 12, 18],
                    [12, 11, 11, 10, 16, 12, 18], [12, 11, 11, 10, 16, 12, 18], [12, 11, 11, 10, 16, 12, 18],
                    [12, 11, 11, 10, 16, 12, 18], [12, 11, 11, 10, 16, 12, 18]]
    yyc=[8.626272750938591, 9.553467006931834, 10.112501036599074, 10.387147491343523, 10.027923228090662,
     9.824906665981647, 10.553643191169042]
    (Wy,W,U,Wz,Uz,Wr,Ur,w,pianchabu)=GRUtrain(traindata1,zhenshidata1,7)
    (Wy, W, U, Wz, Uz, Wr, Ur, w, pianchabu, error2)=GRUtrain2(traindata1,zhenshidata1,7,Wy,W,U,Wz,Uz,Wr,Ur,w)
    print(Wy)
    print(U)
'''

