import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import json
import math
from functools import partial

from config import cfg
from tfflat.base import ModelDesc

from nets.basemodel import resnet50, resnet101, resnet152, resnet_arg_scope, resnet_v1
resnet_arg_scope = partial(resnet_arg_scope, bn_trainable=cfg.bn_train)


def _conv(inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv'):
	with tf.name_scope(name):
		# Kernel for convolution, Xavier Initialisation
		kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
		conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
		return conv
        
def _attention_iter(inputs, lrnSize, itersize, name = 'attention_iter'):
		with tf.name_scope(name):
			numIn = inputs.get_shape().as_list()[3]
			padding = np.floor(lrnSize/2)
			pad = tf.pad(inputs, np.array([[0,0],[1,1],[1,1],[0,0]]))
			U = _conv(pad, filters=1, kernel_size=3, strides=1)
			pad_2 = tf.pad(U, np.array([[0,0],[padding,padding],[padding,padding],[0,0]]))
			sharedK = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([lrnSize,lrnSize, 1, 1]), name= 'shared_weights')
			Q = []
			C = []
			for i in range(itersize):
				if i ==0:
					conv = tf.nn.conv2d(pad_2, sharedK, [1,1,1,1], padding='VALID', data_format='NHWC')
				else:
					conv = tf.nn.conv2d(Q[i-1], sharedK, [1,1,1,1], padding='SAME', data_format='NHWC')
				C.append(conv)
				Q_tmp = tf.nn.sigmoid(tf.add_n([C[i], U]))
				Q.append(Q_tmp)
			stacks = []
			for i in range(numIn):
				stacks.append(Q[-1]) 
			pfeat = tf.multiply(inputs,tf.concat(stacks, axis = 3) )
		return pfeat
	
def _attention_part_crf(inputs, lrnSize, itersize, usepart, name = 'attention_part'):
	with tf.name_scope(name):
		if usepart == 0:
			return _attention_iter(inputs, lrnSize, itersize)
		else:
			partnum = 17
			pre = []
			for i in range(partnum):
				att = _attention_iter(inputs, lrnSize, itersize)
				pad = tf.pad(att, np.array([[0,0],[0,0],[0,0],[0,0]]))
				s = _conv(pad, filters=1, kernel_size=1, strides=1)
				pre.append(s)
			return tf.concat(pre, axis = 3)
        
class Model(ModelDesc):
     
    def head_net(self, blocks, is_training, trainable=True):
        
        normal_initializer = tf.truncated_normal_initializer(0, 0.01)
        msra_initializer = tf.contrib.layers.variance_scaling_initializer()
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        
        with slim.arg_scope(resnet_arg_scope(bn_is_training=is_training)):
            
            out = slim.conv2d_transpose(blocks[-1], 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up1')
            print(out.shape)
            
            att1 = _attention_part_crf(out,1,3,0)
            upsample1 = tf.image.resize_nearest_neighbor(att1, tf.shape(att1)[1:3]*4, name = 'upsampling')
            print(upsample1.shape)
            
            out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up2')
            print(out.shape)
            
            att2 = _attention_part_crf(out,1,3,0)
            upsample2 = tf.image.resize_nearest_neighbor(att1, tf.shape(att2)[1:3]*2, name = 'upsampling')
            print(upsample2.shape)
            
            out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up3')
            
            
            print(out.shape)
            out = _attention_part_crf(out,1,3,0)
            aggatt = tf.add_n([upsample1,upsample2,out])
            print(aggatt.shape)
            
            print("Agg attention shape",aggatt.shape)
            out = _attention_part_crf(aggatt,1,3,1)
            print("Final Output shape", out.shape)
            
            out = slim.conv2d(out, cfg.num_kps, [1, 1],
                    trainable=trainable, weights_initializer=msra_initializer,
                    padding='SAME', normalizer_fn=None, activation_fn=None,
                    scope='out')

        return out
   
    def render_gaussian_heatmap(self, coord, output_shape, sigma):
        
        x = [i for i in range(output_shape[1])]
        y = [i for i in range(output_shape[0])]
        xx,yy = tf.meshgrid(x,y)
        xx = tf.reshape(tf.to_float(xx), (1,*output_shape,1))
        yy = tf.reshape(tf.to_float(yy), (1,*output_shape,1))
              
        x = tf.floor(tf.reshape(coord[:,:,0],[-1,1,1,cfg.num_kps]) / cfg.input_shape[1] * output_shape[1] + 0.5)
        y = tf.floor(tf.reshape(coord[:,:,1],[-1,1,1,cfg.num_kps]) / cfg.input_shape[0] * output_shape[0] + 0.5)

        heatmap = tf.exp(-(((xx-x)/tf.to_float(sigma))**2)/tf.to_float(2) -(((yy-y)/tf.to_float(sigma))**2)/tf.to_float(2))

        return heatmap * 255.
   
    def make_network(self, is_train):
        if is_train:
            image = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.input_shape, 3])
            target_coord = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_kps, 2])
            valid = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_kps])
            self.set_inputs(image, target_coord, valid)
        else:
            image = tf.placeholder(tf.float32, shape=[None, *cfg.input_shape, 3])
            self.set_inputs(image)

        backbone = eval(cfg.backbone)
        resnet_fms = backbone(image, is_train, bn_trainable=True)
        heatmap_outs = self.head_net(resnet_fms, is_train)
        
        if is_train:
            gt_heatmap = tf.stop_gradient(self.render_gaussian_heatmap(target_coord, cfg.output_shape, cfg.sigma))
            valid_mask = tf.reshape(valid, [cfg.batch_size, 1, 1, cfg.num_kps])
            loss = tf.reduce_mean(tf.square(heatmap_outs - gt_heatmap) * valid_mask)
            self.add_tower_summary('loss', loss)
            self.set_loss(loss)
        else:
            self.set_outputs(heatmap_outs)

