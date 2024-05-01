import numpy as np

def my_conv(image,kernel,stride,padding):
    img_batch,img_channel,img_height,img_width = image.shape
    ker_num,ker_channel,ker_height,ker_width = kernel.shape

    out_height = (stride+img_height+2*padding-ker_height)//stride
    out_width = (stride+img_width+2*padding-ker_width)//stride

    out=np.zeros((img_batch,ker_num,out_height,out_width))
    image_padding = np.zeros((img_batch,img_channel,img_height+2*padding,img_width+2*padding))
    image_padding[:,:,padding:padding+img_height,padding:padding+img_width]=image

    for f in range(ker_num):
        for i in range(out_height):
            for j in range(out_width):
                out[:,f,i,j]=np.sum(image_padding[:,:,i*stride:i*stride+ker_height,j*stride:j*stride+ker_width]*
                                    kernel[f,:,:,:],axis=(1,2,3))
    return out

if __name__=="__main__":
    stride = 1
    padding = 0
    x_shape = (1, 1, 4, 4)
    w_shape = (1, 1, 2, 2)
    x = np.ones(x_shape)
    w = np.ones(w_shape)
    print("————image————\n",x)
    print(x.shape)
    print("————kernel————\n",w)
    print(w.shape)
    out = my_conv(x, w,stride,padding)
    print("————output:————\n",out)
    print(out.shape)

