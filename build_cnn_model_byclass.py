# -*- coding: utf-8 -*-
"""
作者：Shayne
程式簡介：使用 class 方式構建卷積網路
"""

import cv2
import tensorflow as tf

def read_and_decode(filename, BATCH_SIZE, MAX_EPOCH): 
    # 建立文件名隊列
    filename_queue = tf.train.string_input_producer([filename], 
                                                    num_epochs=MAX_EPOCH)
    
    # 數據讀取器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    # 數據解析
    img_features = tf.parse_single_example(
            serialized_example,
            features={ 'Label'    : tf.FixedLenFeature([], tf.int64),
                       'image_raw': tf.FixedLenFeature([], tf.string), })
    
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [42, 42])
    
    label = tf.cast(img_features['Label'], tf.int64)

    # 依序批次輸出 / 隨機批次輸出
    # tf.train.batch / tf.train.shuffle_batch
    image_batch, label_batch =tf.train.shuffle_batch(
                                 [image, label],
                                 batch_size = BATCH_SIZE,
                                 capacity = 1000 + 3 * BATCH_SIZE,
                                 min_after_dequeue = 1000)

    return image_batch, label_batch

class myCNN:
    
    def __init__(self, LABEL_NUM):
        # Hyperparameters
        self.LABEL_NUM  = LABEL_NUM
        self.sess = tf.Session()
        
        # Input images & labels
        self.x  = tf.placeholder(tf.float32, shape = [None, 42, 42, 1])
        self.y_ = tf.placeholder(tf.float32, shape = [None, self.LABEL_NUM])
        self.drop_prop = tf.placeholder(tf.float32)
        
    def weight_variable(self, shape, mean=0, stddev=1):
        initial = tf.truncated_normal(shape, mean=mean, stddev=stddev)
        return tf.Variable(initial)
    
    def bias_variable(self, shape, mean=0, stddev=1):
        initial = tf.truncated_normal(shape, mean=mean, stddev=stddev)
        return tf.Variable(initial)
    
    def max_pool_2x2(self, x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='SAME')
    
    def conv2d(self, x, W, strides=[1, 1, 1, 1]):
        return tf.nn.conv2d(x, W, strides=strides, padding='SAME') 
    
    def build(self):
        # Layers
        # Conv1
        self.W_conv1 = self.weight_variable([5, 5, 1, 16])
        self.b_conv1 = self.bias_variable([16])
        self.y_conv1 = self.conv2d(self.x, self.W_conv1) + self.b_conv1
        # ReLU1
        self.relu1 = tf.nn.relu(self.y_conv1)
        # Pool1
        self.pool1 = self.max_pool_2x2(self.relu1)    
        # Conv2
        self.W_conv2 = self.weight_variable([3, 3, 16, 32])
        self.b_conv2 = self.bias_variable([32])
        self.y_conv2 = self.conv2d(self.pool1, self.W_conv2) + self.b_conv2
        # ReLU2
        self.relu2 = tf.nn.relu(self.y_conv2)      
        # Pool2
        self.pool2 = self.max_pool_2x2(self.relu2)
        # FC1
        self.W_fc1 = self.weight_variable([11*11*32, 128])
        self.b_fc1 = self.bias_variable([128])
        self.h_flat = tf.reshape(self.pool2, [-1, 11*11*32])
        self.y_fc1 = tf.matmul(self.h_flat, self.W_fc1) + self.b_fc1
        # ReLU3
        self.relu3 = tf.nn.relu(self.y_fc1)
        # dropout
        self.drop = tf.nn.dropout(self.relu3, self.drop_prop)
        # FC2
        self.W_fc2 = self.weight_variable([128, self.LABEL_NUM])
        self.b_fc2 = self.bias_variable([self.LABEL_NUM])
        self.y  = tf.matmul(self.drop, self.W_fc2) + self.b_fc2
        
        # for predict
        self.y1 = tf.matmul(self.relu3, self.W_fc2) + self.b_fc2
        self.y_pred = tf.argmax(tf.nn.softmax(self.y1), 1)
        
    def train(self, train_x, train_y):
        
        # Cost optimizer
        lossFcn = tf.nn.softmax_cross_entropy_with_logits_v2
        cost = tf.reduce_mean(lossFcn(labels=self.y_, logits=self.y))
    
        train_step = tf.train.AdamOptimizer(0.001).minimize(cost)
        
        correct_prediction = tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

        i = 0
        try:
            while not coord.should_stop():
                image_train, label_train = self.sess.run([train_x, train_y])
                self.sess.run(train_step, feed_dict={self.x : image_train, 
                                                     self.y_: label_train, 
                                                     self.drop_prop: 0.5})
                
                if i % 50 == 0:
                    train_accuracy = self.sess.run(accuracy, feed_dict={
                                                        self.x  : image_train, 
                                                        self.y_ : label_train,
                                                        self.drop_prop: 1.0})

                    print('Iter %d, accuracy %4.2f%%' % (i,train_accuracy*100))
                i += 1
                
        except tf.errors.OutOfRangeError:
            print('Done!')
            
        finally:
            coord.request_stop()
                
        coord.join(threads)

    # 存檔
    def save(self, save_path):
        self.saver = tf.train.Saver()
        tf.add_to_collection('x', self.x)
        tf.add_to_collection('y', self.y_pred)
        self.saver.save(self.sess, save_path)
        
    # 重建
    def restore(self, model_path):
        saver = tf.train.import_meta_graph(model_path+".meta")
        saver.restore(self.sess, model_path)
        self.x = tf.get_collection('x')[0]
        self.y_pred = tf.get_collection('y')[0]
    
    # 預測 
    def predict(self, img):
        img = img.reshape(-1,42,42,1)
        result = self.sess.run(self.y_pred, feed_dict={self.x: img})
        return result

# ========================================================================#
#                              主程式區                                   #
# ========================================================================#
        
filename = './py_Train.tfrecords'
BATCH_SIZE = 128
MAX_EPOCH = 20
LABEL_NUM = 10

# feed data
img_bat, lb_bat = read_and_decode(filename, BATCH_SIZE, MAX_EPOCH)
train_x = tf.reshape(img_bat, [-1, 42, 42, 1])
train_y = tf.one_hot(lb_bat, LABEL_NUM)

# 建模
Model = myCNN(LABEL_NUM)

Model.build()
Model.train(train_x, train_y)

I = cv2.imread('./Dataset/9/img18.jpg')
A = Model.predict(I[:,:,0]/255)
print(A)

# 存檔
Model.save('./model/test_model')

# 重新讀取檔案建模
Model1 = myCNN(LABEL_NUM)
Model1.restore('./model/test_model')

B = Model1.predict(I[:,:,0]/255)
print(B)
