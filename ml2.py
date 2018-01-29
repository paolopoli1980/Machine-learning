#Neural network and the turtle
#dx->w1
       #(right or no)
#dy->w2

#dx->w3
       #(left or no)
#dy->w4

#dx->w5
       #(down or no)
#dy->w6

#dx->w7
       #(up or no)
#dy->w8
import math
import random

def neuron(w1,w2):
    if w1+w2>1:
        res=1
    else:
        res=0
    return res

xhome= [10,10]
xstart=[0,0]
w=[0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75]
wold=[]
wold[:]=w[:]
nsteps=xhome[0]+xhome[1]
i=0
j=0
metric=0
dmem=2000000
dx=xstart[0]-xhome[0]
dy=xstart[1]-xhome[1]
niter=200
nsteps=20
for i in range(niter):
    xhome= [10,10]
    xstart=[0,0]
    cont=0
    i=0
    j=0
    
    while cont<nsteps:
            
        print xstart        
        if neuron(w[i],w[i+1])==1:
           if j==0:
               xstart[0]+=1
               cont+=1
        if neuron(w[i],w[i+1])==1:
           if j==1:
               xstart[0]-=1
               cont+=1
        if neuron(w[i],w[i+1])==1:
           if j==2:
               xstart[1]+=1
               cont+=1
        if neuron(w[i],w[i+1])==1:
           if j==3:
               xstart[1]-=1
               cont+=1
 
        i+=2
        j+=1
        if j>3:
           j=0
           i=0
                  
    metric=math.sqrt((xhome[0]-xstart[0])**2+(xhome[1]-xstart[1])**2)
    if metric<=dmem  and dmem!=0 :
        wold[:]=w[:]
        dmem=metric
    if dmem!=0:    
        for t in range(len(w)):
            casual=random.uniform(-0.2,0.2)
            w[t]=wold[t]+casual
    print xstart[0],xstart[1]
print w
        

 
        

