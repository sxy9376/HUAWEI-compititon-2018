#coding=utf-8
import re
import datetime
import time
import math
import random
from itertools import chain
def predict_vm (ecs_lines , input_lines) :
    # Do your work from here#
    result = []
    if ecs_lines is None:
        print('ecs information is none')
        return result
    if input_lines is None:
        print('input file information is none')
        return result
    temp = int(input_lines[2].replace("\r\n",""))  # 有效规格的个数
    refer_vmflavor = []
    refer_vmcpu = []
    refer_vmmem = []
    refer_iscpuormem = input_lines[-4].replace("\r\n", "")
    for i in range(temp):
        another_temp = input_lines[3 + i]
        another_temp = re.split(" ", another_temp)
        refer_vmflavor.append(another_temp[0])
        refer_vmcpu.append(another_temp[1])
        refer_vmmem.append(another_temp[2].replace("\r", "").replace("\n", ""))
    print("input_lines中的flavor:")
    print(refer_vmflavor)
    print("input_lines中的cpu:")
    print(refer_vmcpu)
    print("input_lines中的mem:")
    print(refer_vmmem)
    print("input_lines中需要预测cpu还是mem:")
    print(refer_iscpuormem)
    print(type(refer_iscpuormem))
    host = re.split(" ", input_lines[0])
    host[2] = host[2].replace("\n", "")
    host_cpu = int(host[0])#物理服务器的cpu
    host_mem = int(host[1])#物理服务器的mem
    pretime_start=time.strptime(input_lines[-2][:10],"%Y-%m-%d")
    pretime_start=datetime.datetime(pretime_start[0],pretime_start[1],pretime_start[2])
    pretime_end=time.strptime(input_lines[-1][:10],"%Y-%m-%d")
    pretime_end=datetime.datetime(pretime_end[0],pretime_end[1],pretime_end[2])
    data_split=[]
    #清空文本文件中的格式
    for line in ecs_lines:
        line=line.replace("\n","")
        line_split=re.split("\t",line)
        data_split.append(line_split)
    #将源文件虚拟机编号用数字替代
    for i in range(len(data_split)):
        data_split[i][2] = data_split[i][2][:10]
        #timestamp = time.mktime(timeArray)
        #data_split[i][2]=timeArray
        #data_split[i][2]=data_split[i][2]/max(list)
        data_split[i][1]=int(data_split[i][1].replace("flavor",""))
        #timeArray = time.strptime(data_split[i][2], "%Y-%m-%d")
        #data_split[i][2] = timeArray
    #对文本文件中的时间序列进行处理,输出为以开始时间,以天数为单位的整形
    date = data_split[0][2]
    date = time.strptime(date, "%Y-%m-%d")
    date = datetime.datetime(date[0], date[1], date[2])
    date_end=time.strptime(data_split[-1][2],"%Y-%m-%d")
    date_end=datetime.datetime(date_end[0],date_end[1],date_end[2])
    t_start=(pretime_start-date_end).days
    t_end=(pretime_end-date_end).days
    for i in range(1,len(data_split)):
        date2=data_split[i][2]
        date2 = time.strptime(date2, "%Y-%m-%d")
        date2 = datetime.datetime(date2[0], date2[1], date2[2])
        if date2==date:
            data_split[i][2]=0
        else:
            data_split[i][2]=(date2-date).days

    for item in data_split:
        del item[0]
    data_split[0][1]=0#将第一天设置为0
    for i in range(1):
        data_1=[]
        data_2=[]
        data_3=[]
        data_4=[]
        data_5=[]
        data_6=[]
        data_7=[]
        data_8=[]
        data_9=[]
        data_10=[]
        data_11=[]
        data_12=[]
        data_13=[]
        data_14=[]
        data_15=[]
    #取出相同VM的申请信息单独保存
    for line in data_split:
        if line[0]==1:
            data_1.append(line)
        elif line[0]==2:
            data_2.append(line)
        elif line[0]==3:
            data_3.append(line)
        elif line[0]==4:
            data_4.append(line)
        elif line[0]==5:
            data_5.append(line)
        elif line[0] == 6:
            data_6.append(line)
        elif line[0] == 7:
            data_7.append(line)
        elif line[0] == 8:
            data_8.append(line)
        elif line[0] == 9:
            data_9.append(line)
        elif line[0] == 10:
            data_10.append(line)
        elif line[0] == 11:
            data_11.append(line)
        elif line[0] == 12:
            data_12.append(line)
        elif line[0] == 13:
            data_13.append(line)
        elif line[0] == 14:
            data_14.append(line)
        elif line[0] == 15:
            data_15.append(line)
    date_list = []
    for line in data_split:
        date_list.append(line[1])
    #求出训练数据的最大日期,为将没有申请该VM的日期补充0值
    maxdate = max(date_list)
    for i in range(1):
        data_1 = datesum(data_1, maxdate)
        data_1=nc(data_1)
        data_1 = cumsum(data_1, 2)
        data_2 = datesum(data_2, maxdate)
        data_2=nc(data_2)
        data_2 = cumsum(data_2, 2)
        data_3 = datesum(data_3, maxdate)
        data_3=nc(data_3)
        data_3 = cumsum(data_3, 2)
        data_4 = datesum(data_4, maxdate)
        data_4=nc(data_4)
        data_4 = cumsum(data_4, 2)
        data_5 = datesum(data_5, maxdate)
        data_5=nc(data_5)
        data_5 = cumsum(data_5, 2)
        data_6 = datesum(data_6, maxdate)
        data_6=nc(data_6)
        data_6 = cumsum(data_6, 2)
        data_7 = datesum(data_7, maxdate)
        data_7=nc(data_7)
        data_7 = cumsum(data_7, 2)
        data_8 = datesum(data_8, maxdate)
        data_8=nc(data_8)
        data_8 = cumsum(data_8, 2)
        data_9 = datesum(data_9, maxdate)
        data_9=nc(data_9)
        data_9 = cumsum(data_9, 2)
        data_10 = datesum(data_10, maxdate)
        data_10=nc(data_10)
        data_10 = cumsum(data_10, 2)
        data_11 = datesum(data_11, maxdate)
        data_11=nc(data_11)
        data_11 = cumsum(data_11, 2)
        data_12 = datesum(data_12, maxdate)
        data_12=nc(data_12)
        data_12 = cumsum(data_12, 2)
        data_13 = datesum(data_13, maxdate)
        data_13=nc(data_13)
        data_13 = cumsum(data_13, 2)
        data_14 = datesum(data_14, maxdate)
        data_14=nc(data_14)
        data_14 = cumsum(data_14, 2)
        data_15 = datesum(data_15, maxdate)
        data_15=nc(data_15)
        data_15 = cumsum(data_15, 2)
    for line in refer_vmflavor:
        if line=="flavor1":
            result.append(line+" "+str(exp_sm(data_1,t_start,t_end)))
        elif line=="flavor2":
            result.append(line+" "+str(exp_sm(data_2,t_start,t_end)))
        elif line == "flavor3":
            result.append(line + " " + str(exp_sm(data_3, t_start,t_end)))
        elif line=="flavor4":
            result.append(line+" "+str(exp_sm(data_4,t_start,t_end)))
        elif line=="flavor5":
            result.append(line+" "+str(exp_sm(data_5,t_start,t_end)))
        elif line=="flavor6":
            result.append(line+" "+str(exp_sm(data_6,t_start,t_end)))
        elif line=="flavor7":
            result.append(line+" "+str(exp_sm(data_7,t_start,t_end)))
        elif line=="flavor8":
            result.append(line+" "+str(exp_sm(data_8,t_start,t_end)))
        elif line=="flavor9":
            result.append(line+" "+str(exp_sm(data_9,t_start,t_end)))
        elif line=="flavor10":
            result.append(line+" "+str(exp_sm(data_10,t_start,t_end)))
        elif line=="flavor11":
            result.append(line+" "+str(exp_sm(data_11,t_start,t_end)))
        elif line=="flavor12":
            result.append(line+" "+str(exp_sm(data_12,t_start,t_end)))
        elif line=="flavor13":
            result.append(line+" "+str(exp_sm(data_13,t_start,t_end)))
        elif line=="flavor14":
            result.append(line+" "+str(exp_sm(data_14,t_start,t_end)))
        elif line=="flavor15":
            result.append(line+" "+str(exp_sm(data_15,t_start,t_end)))
    result.reverse()
    pre_type = []
    vmsum=0
    for line in result:
        vmnum = re.split(" ", line)

        pre_type = pre_type + [[vmnum[0]] * int(vmnum[1])]

        vmsum+=int(vmnum[1])
    put_type = list(chain.from_iterable(pre_type))

    result.reverse()
    result.insert(0,str(vmsum))
    result.append("")

    # 放置
    fact_cpu = int(re.split(' ', input_lines[0])[0])
    fact_mem = int(re.split(' ', input_lines[0])[1]) * 1024  # 换算为MB
    record = []  # 存储放置记录 放置在第几个物理服务器和其型号
    fact_vm_number = 0  # 需要的物理服务器的数量
    sum_mem = 0
    sum_cpu = 0
    # a=-1
    # for i in range(len(put_type)):
    # while int(put_type[i].replace("flavor", "")) < int(refer_vmflavor[a].replace("flavor", "")):
    # a = a - 1
    # sum_mem=sum_mem+int(refer_vmmem[a])
    # sum_cpu=sum_cpu+int(refer_vmcpu[a])
    # print "%%%%%%%%%"
    # print sum_mem*1.0/fact_mem
    # print sum_cpu*1.0/fact_cpu
    dict_cpu = {"flavor1": 1, "flavor2": 1, "flavor3": 1, "flavor4": 2, "flavor5": 2, "flavor6": 2, "flavor7": 4,
                "flavor8": 4, "flavor9": 4, "flavor10": 8, "flavor11": 8, "flavor12": 8, "flavor13": 16, "flavor14": 16,
                "flavor15": 16};
    dict_mem = {"flavor1": 1024, "flavor2": 2048, "flavor3": 4096, "flavor4": 2048, "flavor5": 4096, "flavor6": 8192,
                "flavor7": 4096, "flavor8": 8192, "flavor9": 16384, "flavor10": 8192, "flavor11": 16384,
                "flavor12": 32768, "flavor13": 16384, "flavor14": 32768, "flavor15": 65536}
    # 按cpu分组
    cpu_16 = []
    cpu_8 = []
    cpu_4 = []
    cpu_2 = []
    cpu_1 = []
    random.shuffle(put_type)
    for line in put_type:
        if int(line.replace("flavor", "")) >= 13 and int(line.replace("flavor", "")) <= 15:
            cpu_16.append(line)
        if int(line.replace("flavor", "")) >= 10 and int(line.replace("flavor", "")) <= 12:
            cpu_8.append(line)
        if int(line.replace("flavor", "")) >= 7 and int(line.replace("flavor", "")) <= 9:
            cpu_4.append(line)
        if int(line.replace("flavor", "")) >= 4 and int(line.replace("flavor", "")) <= 6:
            cpu_2.append(line)
        if int(line.replace("flavor", "")) >= 1 and int(line.replace("flavor", "")) <= 3:
            cpu_1.append(line)
    put_type = []
    put_type.append(cpu_16)
    put_type.append(cpu_8)
    put_type.append(cpu_4)
    put_type.append(cpu_2)
    put_type.append(cpu_1)

    while len(put_type[0]) != 0 or len(put_type[1]) != 0 or len(put_type[2]) != 0 or len(put_type[3]) != 0 or len(
            put_type[4]) != 0:

        temp_cpu = fact_cpu
        temp_mem = fact_mem
        fact_vm_number = fact_vm_number + 1

        for i in range(len(put_type)):
            if len(put_type[i]) != 0:
                k = 0
                for j in put_type[i]:
                    if temp_cpu >= dict_cpu[j] and temp_mem >= dict_mem[j]:
                        temp_cpu = dict_cpu[j]
                        temp_mem = temp_mem - dict_mem[j]
                        record.append(str(fact_vm_number))
                        record.append(j)
                        put_type[i][k] = ''
                    k = k + 1

        # "去掉list中的‘’元素"
        for i in range(len(put_type)):
            while '' in put_type[i]:
                put_type[i].remove('')

    # 统计record中各型号出现了几次

    result.append(str(fact_vm_number))  # 输出第一行共需要几台物理服务器
    i = 1  # 控制输出行数

    print len(record)
    #
    while i <= fact_vm_number:
        k = 0  # record下标
        temporary = [0] * 16
        j = 1
        temp_str = str(i)

        while k < len(record):
            if int(record[k]) == i:  # 判断是否部署在第i+1台服务器上
                temporary[int(record[k + 1].replace("flavor", ""))] = temporary[int(
                    record[k + 1].replace("flavor", ""))] + 1  # 用下标记录对应flavor ，下表的内容对应放置个数
            k = k + 2

        while j < 16:  # 去掉没有申请的flavor，添加到result
            if temporary[j] != 0:
                temp_str = temp_str + " " + "flavor" + str(j) + " " + str(temporary[j])
            j = j + 1
        result.append(temp_str)
        i = i + 1
    return result
def cumsum(data,culumn):
    '''
    :param data:
    :param culumn:
    :return: 随日期变化的申请累计值
    '''
    for i in range(1,len(data)):
        data[i][culumn]=data[i-1][culumn]+data[i][culumn]
    return data
def datesum(data,maxdate):
    '''
    将相同申请日期的申请累加,得到该VM在每一天的申请数量和
    '''
    dict={}
    for item in data:
        if item[1] in dict.keys():
            dict[item[1]]+=1
        else:
            dict[item[1]]=1
    for i in range(maxdate+1):
        if i not in dict.keys():
            dict[i]=0
    date=list(dict.keys())
    value=list(dict.values())
    vm=[data[0][0]]*len(date)
    data=[]
    for i in range(len(date)):
        data.append([vm[i]]+[date[i]]+[value[i]])
    data=sorted(data, key=lambda x:x[1])
    return data
def exp_sm(data,t_start,t_end):
    data_test=data[-int((t_end-t_start)*2):]
    data_train=data[:-int((t_end-t_start)*2)]
    alpha=best_expsm_alpha(data_train,data_test)
    s1=[]
    s2=[]
    if len(data)>42:
        s1.append(data[0][2])
    else:
        s1.append((data[0][2]+data[1][2]+data[2][2])/3)
    for i in range(1,len(data)):
        s1.append(alpha*data[i][2]+(1-alpha)*s1[i-1])
    if len(s1)>42:
        s2.append(s1[1])
    else:
        s2.append((s1[0]+s1[1]+s1[2])/3)
    for i in range(1,len(data)):
        s2.append(alpha*s1[i]+(1-alpha)*s2[i-1])
    a=2*s1[-1]-s2[-1]
    b=alpha*(s1[-1]-s2[-1])/(1-alpha)
    t=[]

    return int(round((a+t_end*b)-(a+t_start*b))+6)
def nc(data):
    num=[]
    for line in data:
        num.append(line[2])
    sumdata=sum(num)
    avg=(sumdata*1.0)/len(data)
    err=[]
    for line in data:
        err.append((line[2]-avg)**2)
    err=sum(err)
    err=err/(int(len(data))-1)
    s=math.sqrt(err)
    for i in range(len(data)):
        if data[i][2]>avg+3*s:
             data[i][2]=avg
    return data
def best_expsm_alpha(data_train,data_test):
    test=data_test[-1][2]
    result=[]
    for i in range(1,1000,1):
        alpha=i/1000.0
        s1 = []
        s2 = []
        if len(data_train) > 42:
            s1.append(data_train[0][2])
        else:
            s1.append((data_train[0][2] + data_train[1][2] + data_train[2][2]) / 3)
        for j in range(1, len(data_train)):
            s1.append(alpha * data_train[j][2] + (1 - alpha) * s1[j - 1])
        if len(s1) > 42:
            s2.append(s1[1])
        else:
            s2.append((s1[0] + s1[1] + s1[2]) / 3)
        for k in range(1, len(data_train)):
            s2.append(alpha * s1[k] + (1 - alpha) * s2[k - 1])
        a = 2 * s1[-1] - s2[-1]
        b = alpha * (s1[-1] - s2[-1]) / (1 - alpha)
        pre=round(a+10*b)
        error=abs(pre-test)
        result.append([alpha,error])
    result.sort(key=lambda x:x[1])
    alpha=result[0][0]
    return alpha