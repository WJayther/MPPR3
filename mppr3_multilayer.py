#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mnist import MNIST
from skimage import img_as_ubyte
from skimage.transform import resize
import warnings
import numpy as np
import pickle
from random import shuffle
from tkinter import *

def resized_img(img):
    # resize image, grayscale to black-and-white
    init_img = np.reshape(img,(28,28))
    tmp = str()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tmp = resize(img_as_ubyte(init_img),(14,10))
    tmp[tmp >= 0.5] = 1
    tmp[tmp < 0.5] = 0
    return np.reshape(tmp,140)

def line(x):
    return x

def relu(x):
    return np.maximum(x, 0)

def softplus(x):
    return np.log(1+np.exp(x))

def limit(x,t):
    if x<t: return 0
    return 1
    
def limi_02(x):
    return limit(x,0.2)

def limi_05(x):
    return limit(x,0.5)

def limi_08(x):
    return limit(x,0.8)

def sigma(x,a):
    return 1/(1+np.exp(-a*x))

def sigm_02(x):
    return sigma(x,0.2)

def sigm_05(x):
    return sigma(x,0.5)

def sigm_08(x):
    return sigma(x,0.8)

def sigm_10(x):
    return sigma(x,1)

def htan(x,a):
    pose = np.exp(x*a)
    nege = np.exp(-a*x)
    return (pose-nege)/(pose+nege)

def htan_02(x):
    return htan(x,0.2)

def htan_05(x):
    return htan(x,0.5)

def htan_08(x):
    return htan(x,0.8)

def linlim(x,t):
    if x<0: return 0
    if x<t: return x/t
    return 1

def lili_02(x):
    return linlim(x,0.2)

def lili_05(x):
    return linlim(x,0.5)

def lili_08(x):
    return linlim(x,0.8)


class SimplePerceptron:
    def __init__(self,in_num,value,act=line):
        self.w = (6*np.random.rand(in_num)-3)/10
        self.b = (6*np.random.rand()-3)/10
        self.val = value
        self.act_func = act
    
    def def_func(self,val):
        return val
    
    def ret_prob(self,img):
        return self.act_func(np.sum(self.w * img)+self.b)

class ReluPerceptron(SimplePerceptron):
    def def_func(self,val):
        return relu(val)

class SimpleNLayer:
    def __init__(self,in_num,out_num,act_func=line):
        self.in_num = in_num
        self.out_num = out_num
        self.neurons = list()
        for i in range(out_num):
            self.neurons.append(SimplePerceptron(in_num,i,act_func))
    
    def __getitem__(self,y):
        return self.neurons[y]
    
    def __setitem__(self,y,val):
        self.neurouns[y] = val
    
    def return_y(self,img):
        resp = np.empty(self.out_num)
        for ptron in self.neurons:
            resp[ptron.val]=ptron.ret_prob(img)
        return resp
    
    def train(self,speed):
        prcntg = np.zeros(10, dtype=np.float32)
        totals = np.zeros(10, dtype=np.float32)
        for n in range(len(trn_images)):
            img = trn_images[n]
            cls = trn_labels[n]
            resp = self.return_y(img)
            
            resp_cls = np.argmax(resp)
            resp = np.zeros(self.out_num, dtype=np.float32)
            resp[resp_cls] = 1.0
            
            totals[cls] = totals[cls]+1
            if resp_cls == cls: prcntg[cls]=prcntg[cls]+1
            
            true_resp = np.zeros(self.out_num, dtype=np.float32)
            true_resp[cls] = 1.0
            
            error = resp-true_resp
            
            delta = error * speed * ((resp >= 0) * np.ones(10))
            for i in range(self.out_num):
                self[i].w = self[i].w - np.dot(img,delta[i])
                self[i].b = self[i].b - delta[i]
        tr_str = ''
        for i in range(10):
            tr_str+= '{:0>4.2f} '.format(prcntg[i]/totals[i])
        print(tr_str[:-1])

class MultiLayer:
    def __init__(self,in_num,out_num,act_func=sigm_10):
        self.in_num = in_num
        self.out_num = out_num
        self.hid_num = int(np.round(np.sqrt(in_num*out_num)))
        self.hid_layer = SimpleNLayer(self.in_num,self.hid_num,act_func)
        self.out_layer = SimpleNLayer(self.hid_num,self.out_num,act_func)
    
    def return_y(self,img):
        return self.out_layer.return_y(self.hid_layer.return_y(img))
    
    def train(self,speed):
        prcntg = np.zeros(10, dtype=np.float32)
        totals = np.zeros(10, dtype=np.float32)
        img_indices = list(range(len(trn_images)))
        shuffle(img_indices)
        for n in img_indices[:500]:
            img = trn_images[n]
            cls = trn_labels[n]
            resp = self.return_y(img)
            
            resp_cls = np.argmax(resp)
            resp = 0.1*np.ones(self.out_num, dtype=np.float32)
            resp[resp_cls] = 0.9
            
            totals[cls] = totals[cls]+1
            if resp_cls == cls: prcntg[cls]=prcntg[cls]+1
            
            true_resp = 0.1*np.ones(self.out_num, dtype=np.float32)
            true_resp[cls] = 0.9
            
            error_o = (resp-true_resp) * resp * (1-resp)
            delta_o = error_o * speed
            for i in range(self.out_num):
                self.out_layer[i].w = self.out_layer[i].w - np.dot(self.hid_layer.return_y(img),delta_o[i])
                self.out_layer[i].b = self.out_layer[i].b - delta_o[i]
            
            hid_resp = self.hid_layer.return_y(img)
            error_h = np.zeros(self.hid_num)
            summer = np.zeros(self.hid_num)
            for i in range(self.out_num):
                summer += np.dot(error_o[i],self.out_layer[i].w)
            error_h = summer*hid_resp*(1-hid_resp)
            delta_h = error_h * speed
            for i in range(self.hid_num):
                self.hid_layer[i].w = self.hid_layer[i].w - np.dot(img,delta_h[i])
                self.hid_layer[i].b = self.hid_layer[i].b - delta_h[i]
        tr_str = ''
        for i in range(10):
            tr_str+= '{}:{:0>4.2f} '.format(i,prcntg[i]/totals[i])
        print(tr_str[:-1])

class DrawField:
    def __init__(self):
        self.imgval = list()
        self.checks = list()
        for i in range(140):
            self.imgval.append(IntVar())
            self.checks.append(Checkbutton(root,text='',
               variable=self.imgval[i],onvalue=1,offvalue=0,
               selectcolor='black',bg='white',padx=0,pady=0,
               indicatoron=0,takefocus=0,width=4,height=2))
            self.checks[i].grid(row=(i//10),column=(i%10))
    def getImg(self):
        tmp_return = list()
        for i in self.imgval:
            tmp_return.append(i.get())
        return np.array(tmp_return)
    def purge(self):
        for i in self.checks:
            i.deselect()
    def center_img(self):
        pre_img = self.getImg().tolist()
        if 1 not in set(pre_img): return
        height_top = pre_img.index(1)//10
        height_bot = pre_img[::-1].index(1)//10
        width_left = 9
        width_right = 0
        for i in range(len(pre_img)):
            if pre_img[i] == 1:
                pos = i%10
                if pos>width_right: width_right = pos
                if pos<width_left: width_left = pos
        
        dif_x = int((width_left-(9-width_right))/2)
        if dif_x<0:pre_img.reverse()
        for i in range(abs(dif_x)):
            pre_img.pop(0)
            pre_img.extend([0])
        if dif_x<0:pre_img.reverse()
        
        dif_y = int((height_top-height_bot)/2)
        if dif_y<0: pre_img.reverse()
        for i in range(abs(dif_y)):
            for j in range(10):
                pre_img.pop(0)
                pre_img.extend([0])
        if dif_y<0: pre_img.reverse()
        for i in range(len(pre_img)):
            self.imgval[i].set(pre_img[i])

def c_img(img):
    pre_img = img
    if 1 not in set(pre_img): return pre_img
    height_top = pre_img.index(1)//10
    height_bot = pre_img[::-1].index(1)//10
    width_left = 9
    width_right = 0
    for i in range(len(pre_img)):
        if pre_img[i] == 1:
            pos = i%10
            if pos>width_right: width_right = pos
            if pos<width_left: width_left = pos
        
    dif_x = int((width_left-(9-width_right))/2)
    if dif_x<0:pre_img.reverse()
    for i in range(abs(dif_x)):
        pre_img.pop(0)
        pre_img.extend([0])
    if dif_x<0:pre_img.reverse()
        
    dif_y = int((height_top-height_bot)/2)
    if dif_y<0: pre_img.reverse()
    for i in range(abs(dif_y)):
        for j in range(10):
            pre_img.pop(0)
            pre_img.extend([0])
    if dif_y<0: pre_img.reverse()
    return pre_img

class ComputeButton:
    def __init__(self):
        self.btn = Button(root,text='COMPUTE',
           command=self.compute)
        self.lbl = Label(root,text='',justify=LEFT,anchor=W)
        self.prg = Button(root,text='PURGE',
           command=test.purge)
        self.btn.grid(row=0,column=10)
        self.lbl.grid(row=1,column=10,rowspan=12)
        self.prg.grid(row=13,column=10)
    def compute(self):
        test.center_img()
        out_str = 'PROBABLY: '+str(np.argmax(mlayer.return_y(test.getImg())))+'\n'
        for i in range(10):
            out_str = out_str + str(i) + ':{:>5.2f}\n'
        out_str = out_str[:-1]
        self.lbl.config(text=out_str.format(*mlayer.return_y(test.getImg())))

def print_wb():
    for i in range(10):
        tmp = str()
        wmin = np.min(slayer[i].w)
        wmax = np.max(slayer[i].w)
        for row in range(14):
            for col in range(10):
                nval = slayer[i].w[row*10+col]-wmin
                ooz = int(round(nval/(wmax-wmin),0))
                if ooz == 1:tmp+='X'
                else: tmp+=' '
            tmp+='\n'
        print(tmp)

if __name__ == '__main__':
    mnist_data = MNIST('./mnist_data')
    trn_images, trn_labels = mnist_data.load_training()
    tst_images, tst_labels = mnist_data.load_testing()
    
    #for i in range(0, len(tst_images)):
    #    tst_images[i] = c_img(resized_img(tst_images[i]).tolist())
    
    with open('mnist_test.pickle','rb') as f:
        tst_images = pickle.load(f)
    
    print('FINISHED LOADING: TESTING IMAGES')
    
    #for i in range(0, len(trn_images)):
    #    trn_images[i] = c_img(resized_img(trn_images[i]).tolist())
    
    with open('mnist_train.pickle','rb') as f:
        trn_images = pickle.load(f)
    
    print('FINISHED LOADING: TRAINING IMAGES')
    
    mlayer = MultiLayer(140,10)
    for epoch in range(100):
        mlayer.train(0.2)
        
        total = len(tst_images)
        valid = 0
        invalid = []
        
        for i in range(0, total):
            img = tst_images[i]
            predicted = np.argmax(mlayer.return_y(img))
            true = tst_labels[i]
            if predicted == true:
                valid = valid + 1
        
        print("{} epoch accuracy {}".format(epoch,valid/total))
    
    root = Tk()
    test = DrawField()
    cbtn = ComputeButton()
    #root.mainloop()
