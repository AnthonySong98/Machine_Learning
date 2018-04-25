#0导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE=8
seed=23455

#基于seed生成随机数
rng=np.random.RandomState(seed)
#随机数返回32*2的的矩阵，表示32组，2个特征：体积和重量 作为输入的数据集
X=rng.rand(32,2)
#从X这个32*2的矩阵中 提取一行 判断如果小于1，给Y赋值1，表示合格，否则为0，表示不合格
#作为输入数据的标签
Y=[[int(x0+x1<1)] for (x0,x1) in X]

print("X:\n",X)
print("Y:\n",Y)

#1定义神经网络的输入、参数和输出，定义前向传播的过程
x=tf.placeholder(tf.float32,shape=(None,2))#神经网络的输入
y_=tf.placeholder(tf.float32,shape=(None,1))#y_为预测值,神经网络的输出

#神经网络的隐藏层节点数为3，权向量
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#前向传播
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#2定义损失函数和反向传播的方法
loss=tf.reduce_mean(tf.square(y-y_))#定义损失函数
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)#定义优化方法
#train_step=tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
#train_step=tf.train.AdamOptimizer(0.001).minimize(loss)

#3生成对话，训练STEPS轮
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    #输出目前未经训练的参数取值
    print("w1:\n",sess.run(w1))
    print("w2:\n",sess.run(w2))
    print("\n")
    
    #训练模型
    STEPS=5000
    for i in range(STEPS):
        start=(i*BATCH_SIZE)%32
        end=start+BATCH_SIZE
        #一次喂8个带标签的样本
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if(i%500==0):
            total_loss=sess.run(loss,feed_dict={x:X,y_:Y})
            print("After ",i,"steps,loss is",total_loss)
    
    #输出训练后参数取值
    print("w1:\n",sess.run(w1))
    print("w2:\n",sess.run(w2))
    print("\n")
