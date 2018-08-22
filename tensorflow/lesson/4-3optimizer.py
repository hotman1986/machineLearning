
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:


#载入数据集
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)#当前路径，‘’可以选择存放路径


# In[12]:


#每个批次大小
batch_size = 100
#计算一共有多少个批次
#数据集的数量整除批次大小=多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])#把批次传进去，开始是100，最后一批不一定是多少，我们把平面28*28做成向量传入
y = tf.placeholder(tf.float32,[None,10])#十个标签

#创建一个简单的神经网络
#只用到两个层，784个神经元，输出层为10个神经元
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#定义代价函数
#loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#定义一个求准确率的方法
#如果两个相同就会返回True,不相同就返回False，然后存入correct_prediction
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大值所在的位置

#求准确率
#首先把bool值转化为32为浮点值，然后求平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# In[14]:


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):#把所有的图片训练21次
        for batch in range (n_batch):#一批一批的迭代，一共运行n_batch次
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)#每次获得100张图片
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('Iter'+str(epoch) + ',Testing accuracy'+ str(acc))

