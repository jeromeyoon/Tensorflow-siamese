import random
import numpy as np
import time
import tensorflow as tf 
import input_data
import math

mnist = input_data.read_data_sets("/tmp/data",one_hot=False)

import pdb
def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def mlp(input_,input_dim,output_dim,name="mlp"):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[input_dim,output_dim],tf.float32,tf.random_normal_initializer(mean = 0.001,stddev=0.02))
        return tf.nn.relu(tf.matmul(input_,w))
        
def build_model_mlp(X_,_dropout):

    model = mlpnet(X_,_dropout)
    return model

def mlpnet(image,_dropout):
    l1 = mlp(image,784,128,name='l1')
    l1 = tf.nn.dropout(l1,_dropout)
    l2 = mlp(l1,128,128,name='l2')
    l2 = tf.nn.dropout(l2,_dropout)
    l3 = mlp(l2,128,128,name='l3')
    return l3
def contrastive_loss(y,d):
    tmp= y *tf.square(d)
    #tmp= tf.mul(y,tf.square(d))
    tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
    return tf.reduce_sum(tmp +tmp2)/batch_size/2

def compute_accuracy(prediction,labels):
    return labels[prediction.ravel() < 0.5].mean()
    #return tf.reduce_mean(labels[prediction.ravel() < 0.5])
def next_batch(s,e,inputs,labels):
    input1 = inputs[s:e,0]
    input2 = inputs[s:e,1]
    y= np.reshape(labels[s:e],(len(range(s,e)),1))
    return input1,input2,y
    
# Initializing the variables
init = tf.initialize_all_variables()
# the data, shuffled and split between train and test sets
X_train = mnist.train._images
y_train = mnist.train._labels
X_test = mnist.test._images
y_test = mnist.test._labels
batch_size =128
global_step = tf.Variable(0,trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,10,0.1,staircase=True)
# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(10)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices)
digit_indices = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = create_pairs(X_test, digit_indices)

images_L = tf.placeholder(tf.float32,shape=([None,784]),name='L')
images_R = tf.placeholder(tf.float32,shape=([None,784]),name='R')
labels = tf.placeholder(tf.float32,shape=([None,1]),name='gt')
dropout_f = tf.placeholder("float")

with tf.variable_scope("siamese") as scope:
    model1= build_model_mlp(images_L,dropout_f)
    scope.reuse_variables()
    model2 = build_model_mlp(images_R,dropout_f)

distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(model1,model2),2),1,keep_dims=True))
loss = contrastive_loss(labels,distance)
#contrastice loss
t_vars = tf.trainable_variables()
d_vars  = [var for var in t_vars if 'l' in var.name]
batch = tf.Variable(0)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)
#optimizer = tf.train.RMSPropOptimizer(0.0001,momentum=0.9,epsilon=1e-6).minimize(loss)
# Launch the graph
with tf.Session() as sess:
    #sess.run(init)
    tf.initialize_all_variables().run()
    # Training cycle
    for epoch in range(30):
        avg_loss = 0.
        avg_acc = 0.
        total_batch = int(X_train.shape[0]/batch_size)
        start_time = time.time()
        # Loop over all batches
        for i in range(total_batch):
            s  = i * batch_size
            e = (i+1) *batch_size
            # Fit training using batch data
            input1,input2,y =next_batch(s,e,tr_pairs,tr_y)
            _,loss_value,predict=sess.run([optimizer,loss,distance], feed_dict={images_L:input1,images_R:input2 ,labels:y,dropout_f:0.9})
            feature1=model1.eval(feed_dict={images_L:input1,dropout_f:0.9})
            feature2=model2.eval(feed_dict={images_R:input2,dropout_f:0.9})
            tr_acc = compute_accuracy(predict,y)
            if math.isnan(tr_acc) and epoch != 0:
                print('tr_acc %0.2f' % tr_acc)
                pdb.set_trace()
            avg_loss += loss_value
            avg_acc +=tr_acc*100
        #print('epoch %d loss %0.2f' %(epoch,avg_loss/total_batch))
        duration = time.time() - start_time
        print('epoch %d  time: %f loss %0.5f acc %0.2f' %(epoch,duration,avg_loss/(total_batch),avg_acc/total_batch))
    y = np.reshape(tr_y,(tr_y.shape[0],1))
    predict=distance.eval(feed_dict={images_L:tr_pairs[:,0],images_R:tr_pairs[:,1],labels:y,dropout_f:1.0})
    tr_acc = compute_accuracy(predict,y)
    print('Accuract training set %0.2f' % (100 * tr_acc))

    # Test model
    predict=distance.eval(feed_dict={images_L:te_pairs[:,0],images_R:te_pairs[:,1],labels:y,dropout_f:1.0})
    y = np.reshape(te_y,(te_y.shape[0],1))
    te_acc = compute_accuracy(predict,y)
    print('Accuract test set %0.2f' % (100 * te_acc))
