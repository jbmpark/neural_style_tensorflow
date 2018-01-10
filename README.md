# neural_style_tensorflow
neural style : Image Style Transfer Using Convolutional Neural Networks


A simple implementation of 'Image Style Transfer Using Convolutional Neural Networks' , to see what it does!

Download vgg19 pre-trained : http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
vgg.py is borrowed from 'https://github.com/anishathalye/neural-style'

- Need to fine-tune for hyperparameters ( lr / lr decays... )

bridge input

![입력](./bridge_input.png)

gogh style input

![입력](./style_gogh.jpg) 

gogh-bridge result

![입력](./bridge_gogh.png) 

style_3 input

![입력](./style_3.jpeg)

style_3-bridge

![입력](./bridge_002000.png) 



With multple contents layer losse and total variation loss, it shows very good results!

catsle input

![입력](./catsle_input.png)

catsle-gogh ( conv4_2 * 0.8 , conv2_2 * 0.2 ,  + TV loss )

![입력](./catsle_gogh.png) 



