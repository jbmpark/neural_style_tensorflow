import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import random
import scipy
np.set_printoptions(threshold=np.nan)
import vgg



###
# parameters
GPU='0'
max_tries = 10000    
pooling = 'avg'
feature_size_normalization=False
image_resize = True
tv_loss = True  # total variation loss  from 'image invert'
base_lr = 1.0
contents_loss_ratio = 0.0001
    
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("contents_image", "./catsle.jpg", """Image file path""")
tf.flags.DEFINE_string("style_image", "./style_gogh.jpg", """Image file path""")
tf.flags.DEFINE_string("vgg19", "./imagenet-vgg-verydeep-19.mat", """Pre-trained VGG19 file path""")
#tf.flags.DEFINE_string("invert_layer", "conv5_1", """Layer to invert from.""")
#style_layer=["relu1_2","relu2_2", "relu3_4","relu4_4","relu5_4"] 
style_layer=["conv1_1","conv2_1","conv3_1","conv4_1", "conv5_1"] 
contents_layer=["conv2_2", "conv4_2"]
contents_layer_weight=[0.2, 0.8]
#contents_layer=["conv4_2"]
#contents_layer_weight=[1.0]

sample_dir = './samples_{}'.format(GPU)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

from shutil import copyfile
copyfile(__file__, sample_dir+'/'+__file__)



def read_image(file, scale_w=0, scale_h=0):
    img = scipy.misc.imread(file, mode='RGB').astype(np.float32)
    if (scale_w*scale_h):
        img = scipy.misc.imresize(img, [scale_w, scale_h])
    return img

def gram_matrix(feature):
    gram_matrix = tf.transpose(feature, [0, 3, 1, 2])   # HWC --> CHW
    gram_matrix = tf.reshape(gram_matrix, [tf.shape(feature)[3], tf.shape(feature)[2] * tf.shape(feature)[1]]) # CHW --> [C][H*W]
    gram_matrix = tf.matmul(gram_matrix, gram_matrix, transpose_b=True)
    return gram_matrix



def main(_):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    lr = tf.placeholder(tf.float32, shape=[])

    ### Load pre-trained VGG wieghts
    vgg_mat_file = FLAGS.vgg19
    print ("pretrained-VGG : {}".format(FLAGS.vgg19))
    vgg_weights, vgg_mean_pixel = vgg.load_net(vgg_mat_file)
    print ("vgg_mean_pixel : ", vgg_mean_pixel)
    
    ### Read input image 
    contents_image = FLAGS.contents_image
    style_image = FLAGS.style_image
    print ("contents image : {}".format(contents_image))
    print ("style image    : {}".format(style_image))
    c_img=[]
    s_img=[]
    if image_resize:
        c_img = read_image(contents_image, 224, 224)
        s_img = read_image(style_image, 224, 224)
    else:
        s_img = read_image(style_image)
        c_img = read_image(contents_image)
        #c_img = read_image(contents_image, np.shape(s_img)[0], np.shape(s_img)[1])
        
    
    scipy.misc.imsave(sample_dir+'/0_contents_image.png', c_img)
    scipy.misc.imsave(sample_dir+'/0_style_image.png', s_img)
            
    c_img = c_img - vgg_mean_pixel
    c_img = c_img.astype(np.float32)
    c_img = np.expand_dims(c_img, axis=0)   # extend shape for VGG input
    c_img_shape=np.shape(c_img)
    s_img = s_img - vgg_mean_pixel
    s_img = s_img.astype(np.float32)
    s_img = np.expand_dims(s_img, axis=0)   # extend shape for VGG input
    s_img_shape=np.shape(s_img)
    
    print ("Image shape : ", np.shape(c_img))
    
    with tf.device("/device:GPU:{}".format(GPU)):

        gpu_options = tf.GPUOptions(allow_growth=True)  
        ### Comput content feature of 'invert_layer'
        contents_feature=[]
        style_feature={}
        gram_style_feature=[]
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            c_X = tf.placeholder('float32', shape=c_img_shape)
            s_X = tf.placeholder('float32', shape=s_img_shape)
            c_network = vgg.net_preloaded(vgg_weights, c_X, pooling)
            s_network = vgg.net_preloaded(vgg_weights, s_X, pooling)
            for i in range(len(contents_layer)):
                contents_feature.append(sess.run(c_network[contents_layer[i]], feed_dict={c_X:c_img}))
                #print("contents_feature shape : {}".format(contents_feature.shape))
            for i in range(len(style_layer)):
                style_feature[i] = s_network[style_layer[i]]
                gram_style_feature.append(sess.run(gram_matrix(style_feature[i]), feed_dict={s_X:s_img}))

        
        ### Define network to learn 'X'
        X = tf.Variable(tf.random_normal(c_img_shape))
        X_network = vgg.net_preloaded(vgg_weights, X, pooling)
        

        ### contents feature
        X_contents_feature=[]
        X_contents_feature_shape=[]
        contents_feature_size=[]
        for i in range(len(contents_layer)):
            X_contents_feature.append(X_network[contents_layer[i]])
            X_contents_feature_shape.append(tf.shape(X_contents_feature[i]))
            contents_feature_size.append(tf.cast(X_contents_feature_shape[i][1]*X_contents_feature_shape[i][2]*X_contents_feature_shape[i][3], dtype=tf.float32))
        
        ### contents loss :  l2_loss 
        contents_loss = 0
        for i in range(len(contents_layer)):
            if feature_size_normalization:
                contents_loss = contents_loss + tf.nn.l2_loss(X_contents_feature[i]-contents_feature[i])*contents_layer_weight[i]/contents_feature_size[i]
            else :
                contents_loss = contents_loss + tf.nn.l2_loss(X_contents_feature[i]-contents_feature[i])*contents_layer_weight[i]
        # Can't find in the paper to normalized with feature size, but it could be helpful!

        ### style feature & gram matrix
        X_style_feature={}
        X_gram_style_feature=[]
        for i in range(len(style_layer)):
            X_style_feature[i] = X_network[style_layer[i]]
            X_gram_style_feature.append(gram_matrix(X_style_feature[i]))

        ### style loss 
        gram_loss=[]
        for i in range(len(style_layer)):
            style_feature_shape = tf.cast(tf.shape(X_style_feature[i]), dtype=tf.float32)
            style_feature_size = style_feature_shape[1]*style_feature_shape[2]*style_feature_shape[3]
            gram_diff=X_gram_style_feature[i]-gram_style_feature[i]
            if feature_size_normalization:
                gram_diff=gram_diff/style_feature_size
                
            gram_loss.append( tf.nn.l2_loss(gram_diff) \
                                /(style_feature_shape[3]**2) \
                                /((style_feature_shape[2]*style_feature_shape[1])**2) \
                                /2.0 )
        # L2 loss ( G-A ) : tf.nn.l2_loss(X_gram_style_feature[i]-gram_style_feature[i])
        # N : style_feature_shape[3]
        # M : style_feature_shape[2]*style_feature_shape[1]
        # Style loss in the paper :  (1/(4 * M^2 * N^2))*sum((G-A)^2)
        #                             ==  (1/(2* M^2 * N^2))*l2_loss(G-A)
        #                             ==  l2_loss(G-A) / N^2  / M^2 / 2.0  (this is the style loss definition above!! )

        style_loss = tf.reduce_mean(gram_loss)


        if tv_loss:
            # if image_resize:
            #     total_variation_loss = (tf.image.total_variation(c_img+X)[0] + tf.image.total_variation(s_img+X)[0])/2
            # else:
            #     total_variation_loss = tf.image.total_variation(c_img+X)[0]
            tv_loss_contents = tf.abs(tf.reduce_mean(tf.image.total_variation(tf.convert_to_tensor(c_img)))-tf.reduce_mean(tf.image.total_variation(tf.convert_to_tensor(X))))
            tv_loss_style = tf.abs(tf.reduce_mean(tf.image.total_variation(tf.convert_to_tensor(s_img)))-tf.reduce_mean(tf.image.total_variation(tf.convert_to_tensor(X))))
            total_variation_loss = tv_loss_style
        else:
            total_variation_loss = 0
        loss = contents_loss_ratio*contents_loss + (1-contents_loss_ratio)*style_loss + total_variation_loss
        
        train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step = global_step)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    
    for step in range(max_tries):  
        this_lr = base_lr
        # if step > 1000:
        #     this_lr = base_lr/10.0
        _ , _loss , _c_loss, _s_loss, _tv_loss = sess.run([train_step, loss, contents_loss, style_loss, total_variation_loss], feed_dict={lr:this_lr})      
        print("step: %06d"%step, "loss: {:.04}".format(_loss), "_c_loss: {:.04}".format(contents_loss_ratio*_c_loss), \
            "_s_loss: {:.04}".format((1-contents_loss_ratio)*_s_loss), "_tv_loss: {:.04}".format(_tv_loss))
        

        # testing
        if not (step+1)%100: 
            this_X = sess.run(X)
            this_X = this_X + vgg_mean_pixel
            scipy.misc.imsave(sample_dir+'/styled_{}'.format(str(step+1).zfill(6)) + '.png', this_X[0])
            
    

if __name__ == "__main__":
    tf.app.run()



