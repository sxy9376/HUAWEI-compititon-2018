# coding=utf-8
import math

def xchenlist(x,list1):#数乘向量
    l=len(list1)
    list2=[0 for i in range(l)]
    for i in range(l):
        list2[i]=x*list1[i]
    return list2
def xchenjuzhen(x,list1):#数乘矩阵返回矩阵
    h=len(list1)
    l=len(list1[0])
    list2=[[0 for col in range(l)] for row in range(h)]
    for i in range(h):
        for j in range(l):
            list2[i][j]=x*list1[i][j]
    return list2
def xchulist(x,list1):#数乘向量
    l=len(list1)
    list2=[0 for i in range(l)]
    for i in range(l):
        if(list1[i]==0):
            list2[i]=0
        else:
           list2[i]=float(x)/float(list1[i])
    return list2
def kjianlist(k,list):#1-向量
    l=len(list)
    list1=[0 for i in range(l)]
    for i in range(l):
        list1[i]=k-list[i]
    return list1
def kjialist(k,list):#1-向量
    l=len(list)
    list1=[0 for i in range(l)]
    for i in range(l):
        list1[i]=k+list[i]
    return list1
def listjianlist(list1,list2):


    h = len(list1)
    list3=[0 for i in range(h)]
    for i in range(h):
            list3[i]=list1[i]-list2[i]
    return list3
def juzhenjianjuzhen(list1,list2):
        h = len(list1)
        l = len(list1[0])
        list3 = [[0 for col in range(l)] for row in range(h)]
        for i in range(h):
            for j in range(l):
               list3[i][j] = float(list1[i][j]) - float(list2[i][j])
        return list3

def listchenlist(list1,list2):#点乘
    l=len(list1)
    list3=[0 for i in range(l)]
    for i in range(l):
        list3[i]=list1[i]*list2[i]
    return list3
def listjialist(list1,list2):#点加
    l=len(list1)
    list3=[0 for i in range(l)]
    for i in range(l):
        list3[i]=list1[i]+list2[i]
    return list3
def hangchenlie(list1,list2):#一行n列乘以n行一列，得到一个数
    sumall=0
    l=len(list1)
    for i in range(l):
        sumall=sumall+float(list1[i]*list2[i])
    return sumall
def liechenhang(list1,list2):#n行一列乘以一行n列，得到一个矩阵
    l=len(list1)
    list3=[[0 for col in range(l)] for row in range(l)]
    for i in range(l):
        for j in range(l):
            list3[i][j]=list1[i]*list2[j]

    return list3
def hangchenjuzhen(list1,list2):#一行n列乘以n行n列，得到一行n列
    l=len(list1)
    list3=[0 for i in range(l)]
    for i in range(l):
        for j in range(l):
            list3[i]=list3[i]+list1[i]*list2[i][j]

    return list3
def juzhenchenjuzhen(list1,list2):
    m1=len(list1)
    n=len(list1[0])
    n1=len(list2[0])
    list3=[[0 for col in range(n1)] for row in range(m1)]
    for i in range(m1):
        for j in range(n1):
            for k in range(n):
                list3[i][j] = list3[i][j] + list1[i][k] * list2[k][j]
    return list3

def sigmoid(x):
    if(type(x)!=list):
        output= 1/(1+pow(math.e,-x))
    else:
        l=len(x)
        output=[0.0 for i in range(l)]
        for i in range(l):
            output[i]=1/(1+pow(math.e,-float(x[i])))
    return output
def sigmoidtoD(x):
    if (type(x) != list):
        output = 1 / (1 + pow(math.e, -x))
        output=output*(1-output)
    else:
        l = len(x)
        output = [0.0 for i in range(l)]
        for i in range(l):
            output[i] = 1 / (1 + pow(math.e, -float(x[i])))
            output[i] = output[i] * (1 - output[i])

    return output
# convert output of sigmoid function to its derivative
#求西格玛函数的导数在x处
def Relu(x):
    if (type(x) != list):
        if (x <= 0):
            output = 0.0
        else:
            output = float(x)
    else:
        l = len(x)
        output = [0.0 for i in range(l)]
        for i in range(l):
            if (x[i] <= 0):
                output[i] = 0.0
            else:
                output[i] = float(x[i])
    return output
def RelutoD(x):
    if (type(x) != list):
        if (x <= 0):
            output = 0.0
        else:
            output = 1
    else:
        l = len(x)
        output = [0.0 for i in range(l)]
        for i in range(l):
            if (x[i] <= 0):
                output[i] = 0.0
            else:
                output[i] = 1
    return output
def tanh(x):
    if (type(x) != list):
        a = pow(math.e, x)
        b = pow(math.e, -x)
        xx = (a - b) / (a + b)
        output = xx
    else:
        l = len(x)
        output = [0.0 for i in range(l)]
        for i in range(l):
            a = pow(math.e, float(x[i]))
            b = pow(math.e, -float(x[i]))
            output[i] = (a - b) / (a + b)
    return output
def tanhtoD(x):
    if (type(x) != list):
        a = pow(math.e, x)
        b = pow(math.e, -x)
        xx = (a - b) / (a + b)
        output = 1-pow(xx,2)
    else:
        l = len(x)
        output = [0.0 for i in range(l)]
        for i in range(l):
            a = pow(math.e, float(x[i]))
            b = pow(math.e, -float(x[i]))
            xx = (a - b) / (a + b)
            output[i]=1-pow(xx,2)
    return output
def gettraindata(list, k):
    listtraindata = []
    listyucedata = []
    Num = len(list)-1
    M = (len(list)-1)/ k  # M表示list可以分成多少组
    for i in range(M):

        listtraindata.append(list[(Num - k * (i + 1)):(Num - k * i)])
        listyucedata.append(list[(Num + 1 - k * (i + 1)):(Num + 1 - k * i)])
    listtraindata.reverse()
    listyucedata.reverse()
    # print listtraindata
    # print listyucedata

    return listtraindata, listyucedata
def gundong(list,x):
    l=len(list)
    list1=[0 for i in range(l)]
    for i in range(l-1):
        list1[i]=list[i+1]
    list1[l-1]=x
    return list1
if __name__ == "__main__":
    list1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    a,b = gettraindata(list1,8)
    print a
    print b
