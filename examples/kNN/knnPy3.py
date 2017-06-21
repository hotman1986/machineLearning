# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:59:00 2017

@author: Administrator
"""

import csv
import random
import math
import operator
#装载文件
def loadDataset(filename,split,trainingSet=[],testSet=[]):#在数据集中有一部分作为训练集，有一部分作为测试集，split表示分类间断点
    with open(filename,'r') as csvfile:#装载为csvfile
        lines = csv.reader(csvfile)#读取所有的行
        dataset = list(lines)#装换成list的数据结构
        for x in range(len(dataset)-1):
            for y in range(4):#元组长度为4，因为有四个特征
                dataset[x][y] = float(dataset[x][y])
                if random.random()<split:
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])
            

#欧拉距离
def euclideanDistance(instance1,instance2,length):
    distance =0
    for x in range(length):
        distance+=pow((instance1[x]-instance2[x]),2)
        return math.sqrt(distance)

#返回最近的k个邻居
def getNeighbors(trainingSet,testInstance,k):#一个训练集，testInstace唯一一个训练实例，k个离实例testInstace最近的点
    distances=[]
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance,trainingSet[x],length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

                 
#判断实例并且归类

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]#-1为最后一个值
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
    sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)#reverse降序 operator.itemgetter(1)是取第二个key值
    return sortedVotes[0][0]#投票最大的
            

#预测的准确率    
def getAccuracy(testSet,predictions):
    correct=0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct+=1
        return (correct/float(len(testSet)))*100.0

def main():
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset(r'D:\iris.data.txt',split,trainingSet,testSet)#r表示raw原始数据，忽略符号
    print ('Train set:' + repr(len(trainingSet)))
    print ('Test set:'+ repr(len(testSet)))
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet,testSet[x],k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('>predicted'+repr(result)+', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet,predictions)
    print('Accuracy:'+repr(accuracy)+'%')
    
main()

    
    
