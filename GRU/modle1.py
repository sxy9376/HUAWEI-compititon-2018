# coding=utf-8
import getTraindata
import function
import GRU
import ecs
def modle(list,types,yucetianshu):
    historyfile = 'data201501-05.txt'
    historylins =ecs.read_lines(historyfile)
    Num1 = getTraindata.getTraindata(historylins)
    #print(Num1)
    #print(len(Num1[0]))
    '''
    print(len(Num1[0]))
    kk=0
    wucha=[]
    #print(Num1)
    for danyuanshu in range(3,8):
       (listtraindata, listzhenshidata)=function.gettraindata(Num1[types],danyuanshu)
       (Wy, W, U, Wz, Uz, Wr, Ur, w, pianchabu) = GRU.GRUtrain(listtraindata, listzhenshidata, danyuanshu)
       #print(1)
       #print(0)
       (listtraindata1, listzhenshidata1) = function.gettraindata(list, danyuanshu)
       #print(2)
       #print(danyuanshu)
       (Wy1, W1, U1, Wz1, Uz1, Wr1, Ur1, w1, pianchabu1,error)=GRU.GRUtrain2(listtraindata1,listzhenshidata1,danyuanshu,Wy,W,U,Wz,Uz,Wr,Ur,w)
       #print(3)
       wucha.append(error)
    for i in range(len(wucha)):
        if(wucha[i]==min(wucha)):
            hdim=i+3
            break
    
    '''
    #print(list)
    #print(len(list))
    hdim=12
    (listtraindata2, listzhenshidata2) = function.gettraindata(Num1[types], hdim)
    (Wy2, W2, U2, Wz2, Uz2, Wr2, Ur2, w2, pianchabu2) = GRU.GRUtrain(listtraindata2, listzhenshidata2, hdim)
    (listtraindata3, listzhenshidata3) = function.gettraindata(list, hdim)
    #(Wy3, W3, U3, Wz3, Uz3, Wr3, Ur3, w3, pianchabu3, error3) = GRU.GRUtrain2(listtraindata3, listzhenshidata3, hdim,Wy2, W2, U2, Wz2, Uz2, Wr2, Ur2, w2)
    #预测
    length=len(listzhenshidata3)
    yucedata=listzhenshidata3[length-1]
    #print(yucedata)
    yucezhi=[]
    for i in range(yucetianshu):
        yuce=abs(GRU.GRUyuce(yucedata,hdim,Wy2, W2, U2, Wz2, Uz2, Wr2, Ur2, w2, pianchabu2))
        yucezhi.append(yuce)
        yucedata=function.gundong(yucedata,yuce)
    return sum(yucezhi)
