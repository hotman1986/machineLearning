import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn import preprocessing

allElectronicsData = open(r'D:\1.csv','rb')
reader = csv.reader(allElectronicsData)
headers = reader.next()
print (headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1,len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

#print(featureList)

vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

#print("dummyX"+str(dummyX))
#print(vec.get_feature_names())
#print("LabelList:"+str(labelList))

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
#print("dummyY:"+str(dummyY))

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(dummyX,dummyY)
#print("clf:"+str(clf))

with open("d://aaa.dot",'w') as f:
    #dot -Tpdf allElectronicInformationGainOri.dot -o output.pdf
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)

oneRowX = dummyX[0,:]
newRowX = oneRowX
newRowX[2] = 1
newRowX[0] = 0


print("newRowX: " + str(newRowX))

predictY = clf.predict(newRowX)
print("predictY:" + str(predictY))


