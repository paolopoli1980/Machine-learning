#####################################################################################
######AI, Neural Network game example for an understanding of machine learning#######
#####################################################################################

import pygame
from pygame.locals import *
from sys import exit
import os
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
#####################################################################################

clock = pygame.time.Clock() 

def neuron(mat,numberofneurons,numberofinput,dx,dy,vball,vdotv,tipo,vballx,vbally,updx,dwdx,upsx,dwsx,dsidexdx,dsideyup,dsidexsx,dsideydw,centerdist,vtimesvdotv,numberofneurons_layer2,numbinputfirst,numbinputsec,numbneuronsfirst,numbneuronsecond):

    v=[vballx,vbally,updx,dwdx,upsx,dwsx]
    #v=[vtimesvdotv,vdotv,updx,dwdx,upsx,dwsx]
   # v=[vballx,vbally,updx,dwdx,upsx,dwsx,centerdist]
    
    #v=[vballx,vbally,updx,dwdx,upsx,dwsx,dsidexdx,dsideyup,dsidexsx,dsideydw]
    module=np.sqrt(sum(i*i for i in v))
    v=[i/module for i in v]
#    print (v)
    keymem=[]
###################################Random choice of the agent#####################################    
    if tipo=='random':
        for i in range(numberofneurons):
            sumneuronvar=0
            c=np.random.randint(0,2)
            if c==1:
                keymem.append(i)    
###################################################################################################

########################one layer sigmoide activation function#####################################                
    if tipo=='sigmoide':
        #print (numberofneurons,numberofinput)
        for i in range(numberofneurons):
            sumneuronvar=0
            for j in range(numberofinput):
                #print (mat[j][i])
                sumneuronvar+=mat[j][i]*v[j]
                #print (j,i)

            #print (sumneuronvar)
            valneur=1/(1+np.exp(-sumneuronvar))
           # print ("valneur %s",valneur)            
            var=np.random.uniform()

            if var<valneur:
                    keymem.append(i)    
###################################################################################################

########################one layer heavside activation function#####################################
    if tipo=='heaviside':
       # print (tipo)
        #print (numberofneurons,numberofinput)
        for i in range(numberofneurons):
            sumneuronvar=0
            for j in range(numberofinput):
                #print (mat[j][i])
                sumneuronvar+=mat[j][i]*v[j]
                #print (j,i)

            #print (sumneuronvar)
           # print ("valneur %s",valneur)            
            if sumneuronvar>0:    
                keymem.append(i)
                
    #print(keymem)
   # print (mat)
    if tipo=="eluandsigmo":
        aix=[]
        for i in range(numbneuronsfirst):
            sumneuronvar=0
            for j in range(numbinputfirst):
                sumneuronvar+=mat[j][i]*v[j]
            if sumneuronvar<0:
                sumneuronvar=np.exp(sumneuronvar)-1
            aix.append(sumneuronvar)
        for i in range(numbneuronsfirst,numberofneurons):
            sumneuronvar=0
            for j in range(numbinputsec):
                sumneuronvar+=mat[j][i]*aix[j]
            valneur=1/(1+np.exp(-sumneuronvar))
           # print ("valneur %s",valneur)            
            var=np.random.uniform()

            if var<valneur:
                    keymem.append(i-numbneuronsfirst)    
                
    if tipo=="linesigmosoft":
        aix=[]
        global valsoft
        global sumsoft
        valsoft=[0 for i in range(numbneuronsfirst,numberofneurons)]
        for i in range(numbneuronsfirst):
            sumneuronvar=0
            for j in range(numbinputfirst):
                sumneuronvar+=mat[j][i]*v[j]

            aix.append(sumneuronvar)
        for i in range(numbneuronsfirst,numberofneurons):
            sumneuronvar=0
            for j in range(numbinputsec):
                sumneuronvar+=mat[j][i]*aix[j]
            valneur=1/(1+np.exp(-sumneuronvar))
           # print ("valneur %s",valneur)
            valsoft[i-numbneuronsfirst]=valneur
      #  print (valsoft)            
        sumsoft=0
        for k in range(numbneuronsfirst,numberofneurons):
            sumsoft+=np.exp(valsoft[k-numbneuronsfirst])
        for k in range(numbneuronsfirst,numberofneurons):
            valneur=np.exp(valsoft[k-numbneuronsfirst])/sumsoft
            var=np.random.uniform()

            if var<valneur:
                    keymem.append(k-numbneuronsfirst)            
       # print (valsoft)
       # print (sumsoft)
      #  print (keymem)

    if tipo=="doubleheaviside":
        aix=[]
        for i in range(numbneuronsfirst):
            sumneuronvar=0
            for j in range(numbinputfirst):
                sumneuronvar+=mat[j][i]*v[j]
            if sumneuronvar>0:
                sumneuronvar=1
            else:
                sumneuronvar=0
            aix.append(sumneuronvar)
        for i in range(numbneuronsfirst,numberofneurons):
            sumneuronvar=0
            for j in range(numbinputsec):
                sumneuronvar+=mat[j][i]*aix[j]
            valneur=sumneuronvar
           # print ("valneur %s",valneur)            
            

            if valneur>0:
                    keymem.append(i-numbneuronsfirst)    

    if tipo=="linesoft":
        aix=[]
        global valsoftb
        global sumsoftb
        valsoftb=[0 for i in range(numbneuronsfirst,numberofneurons)]
        for i in range(numbneuronsfirst):
            sumneuronvar=0
            for j in range(numbinputfirst):
                sumneuronvar+=mat[j][i]*v[j]

            aix.append(sumneuronvar)
        for i in range(numbneuronsfirst,numberofneurons):
            sumneuronvar=0
            for j in range(numbinputsec):
                sumneuronvar+=mat[j][i]*aix[j]
            valneur=sumneuronvar
           # print ("valneur %s",valneur)
            valsoftb[i-numbneuronsfirst]=valneur
      #  print (valsoft)            
        sumsoftb=0
        for k in range(numbneuronsfirst,numberofneurons):
            sumsoftb+=np.exp(valsoftb[k-numbneuronsfirst])
        for k in range(numbneuronsfirst,numberofneurons):
            valneur=np.exp(valsoftb[k-numbneuronsfirst])/sumsoftb
            var=np.random.uniform()

            if var<valneur:
                    keymem.append(k-numbneuronsfirst)          
    if tipo=="elusoft":
        aix=[]
        global valsoftc
        global sumsoftc
        valsoftc=[0 for i in range(numbneuronsfirst,numberofneurons)]
        for i in range(numbneuronsfirst):
            sumneuronvar=0
            for j in range(numbinputfirst):
                sumneuronvar+=mat[j][i]*v[j]
            if sumneuronvar<0:
                sumneuronvar=np.exp(sumneuronvar)-1
            aix.append(sumneuronvar)
        for i in range(numbneuronsfirst,numberofneurons):
            sumneuronvar=0
            for j in range(numbinputsec):
                sumneuronvar+=mat[j][i]*aix[j]
            valneur=sumneuronvar

           # print ("valneur %s",valneur)
            valsoftc[i-numbneuronsfirst]=valneur
      #  print (valsoft)            
        sumsoftc=0
        for k in range(numbneuronsfirst,numberofneurons):
            sumsoftc+=np.exp(valsoftc[k-numbneuronsfirst])
        for k in range(numbneuronsfirst,numberofneurons):
            valneur=np.exp(valsoftc[k-numbneuronsfirst])/sumsoftc
            var=np.random.uniform()

            if var<valneur:
                    keymem.append(k-numbneuronsfirst) 

    if tipo=="elusoftmax":
        aix=[]
        global valsoftd
        global sumsoftd
        movek=0
        valsoftd=[0 for i in range(numbneuronsfirst,numberofneurons)]
        for i in range(numbneuronsfirst):
            sumneuronvar=0
            for j in range(numbinputfirst):
                sumneuronvar+=mat[j][i]*v[j]
            if sumneuronvar<0:
                sumneuronvar=np.exp(sumneuronvar)-1
            aix.append(sumneuronvar)
        for i in range(numbneuronsfirst,numberofneurons):
            sumneuronvar=0
            for j in range(numbinputsec):
                sumneuronvar+=mat[j][i]*aix[j]
            valneur=sumneuronvar

           # print ("valneur %s",valneur)
            valsoftd[i-numbneuronsfirst]=valneur
      #  print (valsoft)            
        sumsoftd=0
        for k in range(numbneuronsfirst,numberofneurons):
            sumsoftd+=np.exp(valsoftd[k-numbneuronsfirst])
        var=0    
        for k in range(numbneuronsfirst,numberofneurons):
            valneur=np.exp(valsoftd[k-numbneuronsfirst])/sumsoftd
            #var=np.random.uniform()
            if valneur>var:
                var=valneur
                movek=k
        keymem.append(movek-numbneuronsfirst)
        
    if tipo=="elusoftlim":
        aix=[]
        global valsofte
        global sumsofte
        movek=0
        valsofte=[0 for i in range(numbneuronsfirst,numberofneurons)]
        for i in range(numbneuronsfirst):
            sumneuronvar=0
            for j in range(numbinputfirst):
                sumneuronvar+=mat[j][i]*v[j]
            if sumneuronvar<0:
                sumneuronvar=np.exp(sumneuronvar)-1
            aix.append(sumneuronvar)
        for i in range(numbneuronsfirst,numberofneurons):
            sumneuronvar=0
            for j in range(numbinputsec):
                sumneuronvar+=mat[j][i]*aix[j]
            valneur=sumneuronvar

           # print ("valneur %s",valneur)
            valsofte[i-numbneuronsfirst]=valneur
      #  print (valsoft)            
        sumsofte=0
        for k in range(numbneuronsfirst,numberofneurons):
            sumsofte+=np.exp(valsofte[k-numbneuronsfirst])
        var=0.75
        for k in range(numbneuronsfirst,numberofneurons):
            valneur=np.exp(valsofte[k-numbneuronsfirst])/sumsofte
            #var=np.random.uniform()
            if valneur>var:
                #var=valneur
                movek=k
                keymem.append(movek-numbneuronsfirst)
        
    if tipo=="lineheaviside":
        aix=[]
        for i in range(numbneuronsfirst):
            sumneuronvar=0
            for j in range(numbinputfirst):
                sumneuronvar+=mat[j][i]*v[j]

            aix.append(sumneuronvar)
        for i in range(numbneuronsfirst,numberofneurons):
            sumneuronvar=0
            for j in range(numbinputsec):
                sumneuronvar+=mat[j][i]*aix[j]
            valneur=sumneuronvar
           # print ("valneur %s",valneur)
      #  print (valsoft)            
   

            if valneur>=0:
                keymem.append(i-numbneuronsfirst)     

    if tipo=="eluheaviside":
        aix=[]
       
        
        for i in range(numbneuronsfirst):
            sumneuronvar=0
            for j in range(numbinputfirst):
                sumneuronvar+=mat[j][i]*v[j]
            if sumneuronvar<0:
                sumneuronvar=np.exp(sumneuronvar)-1
            aix.append(sumneuronvar)
        for i in range(numbneuronsfirst,numberofneurons):
            sumneuronvar=0
            for j in range(numbinputsec):
                sumneuronvar+=mat[j][i]*aix[j]
            valneur=sumneuronvar

           # print ("valneur %s",valneur)
            
      #  print (valsoft)            
        
            if valneur>=0:
                    keymem.append(i-numbneuronsfirst) 
    if tipo=='manual':
        pass
#####################################################################################################
   
    return keymem        


def ball_settings():
   
    for i in range(nballs):
        r1=(random.randint(-4,4))
        r2=(random.randint(-4,4))
        speed.append([r1,r2])


    
def ballmoving(i):
    
    if ballrect[i].left < 0 or ballrect[i].right > width:
        speed[i][0] = -speed[i][0]
    if ballrect[i].top < 0 or ballrect[i].bottom > height:
        speed[i][1] = -speed[i][1]
 

def drawtheborder(limplayx,limplayy):
    pygame.draw.rect(screen,(0,0,255),(width/2-limplayx,height/2-limplayy,limplayx*2,limplayy*2),2)
    
def total_variational_method(matweight):     

    

    matweight=matweightold+add                          

    print("AdD")
    print(add)
    print (deltamin,deltamax)
   
    return matweight
    
def single_variational_method(numberofinput,numberofneurons):     
    vc=np.random.uniform(deltamin,deltamax)
    i=np.random.randint(0,numberofinput)
    j=np.random.randint(0,numberofneurons)
    matweight[i][j]=matweightold[i][j]+vc                            
    return matweight

def single_ordered_variational_method(memi,memj):
    vc=np.random.uniform(deltamin,deltamax)
    matweight[memi][memj]=matweightold[memi][memj]+vc                            
    return matweight    

def product_variational_method():
    vc=np.random.uniform(deltamin,deltamax)
    #print (vc)
    for i in range(numberofinput):
        for j in range(numberofneurons):
            matweight[i][j]=matweightold[i][j]*vc 
    return matweight    
 
def gradient_method(deltagrad,numberofneurons,numberofinput):

    cgx[0]+=1


    if cgx[0]==numberofneurons:
        cgy[0]+=1
        cgx[0]=0
    if cgy[0]==numberofinput:
        cgy[0]=-1
        cgx[0]=-1
        
    if cgy[0]!=-1:
        matweight[cgy[0]][cgx[0]]+=deltagrad
        startgrad[0]=1
    else:
        startgrad[0]=0
    
def genetic_algorithm(geneticmat,numberofgeneticmatrix,numberofaceptedgeneticmat,numberofinput,numberofneurons,deltamax,deltamin):
    maxim=0
    memindex=0
    indexlist=[]
    matgennew=np.random.rand(numberofgeneticmatrix,numberofinput,numberofneurons)*0    
    print (memgenetictime)
    for t in range(numberofaceptedgeneticmat):
        maxim=0
        for k in range(numberofgeneticmatrix):
            if memgenetictime[k]>maxim:
                maxim=memgenetictime[k]
                memindex=k
        indexlist.append(memindex)
        memgenetictime[memindex]=-1
    print (indexlist)    
    print ("geneticmat")
    print(geneticmat)
    cont=-1
    for el in indexlist:
        print (el)
        cont+=1
        matgennew[cont]=geneticmat[int(el)].copy()
#        print (matgennew[int(el)][y][x])
    print ("matgennew")
    print(matgennew)    
    cont=numberofaceptedgeneticmat-1    
    for j in range(0,numberofaceptedgeneticmat):
        
        for t in range(int(numberofgeneticmatrix/numberofaceptedgeneticmat)-1):
            cont+=1
            for y in range(numberofinput):
                for x in range(numberofneurons):
                    matgennew[cont][y][x]=matgennew[j][y][x]+np.random.uniform(deltamin,deltamax)
    for t in range(cont+1):
        
        geneticmat[t]=matgennew[t]
    print ("geneticmat")
    print(geneticmat)

    return geneticmat        
        
            
        


pygame.init()
size = width, height = 400, 400
speed=[]
screen = pygame.display.set_mode(size)
background = pygame.image.load('Immagine.png').convert()
screen.blit(background, (0, 0))
white = 255, 255, 255
nballs=25
#k=1

nattemps=150 #######nattemps number of attemps you want to do########
nreductionlim=50 #######any nreductionlimit attempted the random decreasing become half########
starttinyincrement=2001 ######after this limit it start with a tiny incrememt#######
timegradientcomparing=0
matrixtimechanging=2001
maxsetmat=10
restart=2002
switchuniformtogaussian=2001  ######after this number steps start the gaussian increment in case of total_method_variation#####
standardev=0.125     #####deviation standard for the gaussian increment############
numberofgeneticmatrix=4
numberofaceptedgeneticmat=2
#k=110
timeclockold=0  #####the longest past time is memorized##########
timeclock=0     #####the time of the n attemping is memorized#######
timeclockgrad=0 #####??????????????????############
gradcoef=0.5      #####is the learning parameter in case you chose the gradient method######      
bestk=0         #####??????????????????????#####
bestradius=0
ballrect=[]
ball=[]
r1=[]
r2=[]
#posball=([0,0])
numberofneurons=3
numberofneurons_layer2=4 #it has to be 4 in case of the single layer is on else it is 0!!!!
numberofinput=6

typeofincrement=''
weightold=[]
cgx=[0]
cgy=[0]
add=[]
np.random.seed() 
limplayx=180      
limplayy=160
deltamin=-0.1#######minimal increment during the back propagation########
deltamax=0.1######maximal increment during the back propagation########
memdeltamin=deltamin
memdeltamax=deltamax
tinydelta=0.00125  #######the tinydelta increment when it start tinydelta modality######
deltagrad=0.25  #######delta variation in gradient method#########
awardingprize='false'
prizecoef=0.98

numbneuronsfirst=numberofneurons
numbneuronsecond=numberofneurons_layer2
numbinputfirst=numberofinput
numbinputsec=numberofneurons

if numberofneurons_layer2>0:
    ntotneurons=numberofneurons+numberofneurons_layer2
    ntotinput=max(numberofinput,numberofneurons)
    numbinputfirst=numberofinput
    numbinputsec=numberofneurons
    numbneuronsfirst=numberofneurons
    numbneuronsecond=numberofneurons_layer2
    numberofinput=ntotinput
    numberofneurons=ntotneurons

s=(numberofinput,numberofneurons)
matweight=np.random.rand(numberofinput,numberofneurons)*maxsetmat-maxsetmat/2

#matweight=np.array([[0,0,0,0],[0,0,0,0],[10,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
#matweight=np.random.rand(numberofinput,numberofneurons)

#elusoftmax nosides
'''
[Elusoftmax hidden layer four neurons]
matweight=[[ 5.34856863e+01, -5.24128522e+01,  7.08371023e+01, -3.41929613e+01,
   -4.50532234e+01,  3.08501902e+01,  2.35263955e+01, -1.10485454e+02,
   -9.73857159e+00],
  [-6.76189468e+01,  9.66178883e+01, -4.77864407e+01, -2.57990029e+01,
    3.83051103e+01,  7.73704936e+01,  6.95143487e+01, -6.05072552e+01,
    7.99549878e+01],
  [-1.43261008e+01,  1.27055035e+00, -1.85888996e+01,  3.82678005e+01,
   -6.27532144e-01, -7.36335946e+01, -1.24873480e+02, -7.08296334e+01,
   -2.83508411e+01],
  [ 1.38753208e+01, -1.68045140e+01,  8.73453200e+01, -4.71877646e+00,
    9.84638442e+01,  3.65602194e+01, -3.65339433e+01, -6.37015609e+01,
   -2.16207217e+01],
  [ 8.79115498e+00, -5.97299614e+01, -1.84253757e+01, -5.18250914e+01,
   -2.71255411e+01, -1.17183167e+01, -2.14448323e+01,  7.09501979e+01,
   -1.38529799e+02],
  [ 2.15108974e+01, -2.60501879e+01, -1.34855997e+02, -1.18659427e+02,
   -6.91923272e+01, -1.01249555e+02, -1.58670836e+01,  3.28419147e+01,
    7.88420837e-01]]



matweight=[[-14.58980297, -24.04309938,   5.53849418, -17.80602909,  -8.42141932,
   -3.24913684,  10.40216906,  -3.59727236],
 [ 10.51042359,  -3.56832696, -30.38040599, 10.61605403, -21.18851438,
   12.98860993,   8.78632697,  23.73331051],
 [ -1.29320148,  -1.92129111,   3.06979549,   0.65546965,  -2.37920229,
    4.14222402,   8.11443802,  16.92479162],
 [-14.0148265,   -4.05176534,  14.74471022,  -9.53353317, -18.25834379,
   -2.36497642,  -7.27693648,   1.79170612],
 [  3.75441862, -19.412406,    -3.4492203,   -4.93446537,  -7.65669514,
   11.30217018,  27.17843471,  12.51225677],
 [-10.64849338,   4.65678499,  23.52703653,  11.29205734,  25.85236905,
    0.22425672+10,  -3.13792134,  -6.00410148]]

[[ 0.64761962 -3.20072042 -3.5837733   2.65868994 -2.64349189  0.17800418
   2.67132602 -4.55847397]
 [-2.77733812 -1.41828906 -4.44315558 -2.91116693  0.94225973 -2.30260472
   0.09206093 -1.12760442]
 [ 2.48716317 -4.37026032 -0.57109528  4.2294066   3.74334977 -1.00684843
  -2.92962425  3.67470599]
 [-4.04493935 -3.14064909  4.57762271 -3.31578502  3.49777968 -1.93249113
   4.27151112  4.48670301]
 [ 4.56898534  1.53849651 -3.54418593  2.19440892 -1.8993946  -0.46252033
   3.40305223  3.66824954]
 [ 3.83726478 -1.50222531 -1.36324965 -3.08230469  4.32021526 -3.6092124
   2.97922564  3.50959319]]

[[-5.81803650e+00  6.51508414e+00 -5.96675534e+00 -6.62313176e-01
  -2.02767939e+01  8.12323548e+00  5.12238499e+00  1.25528293e+01]
 [ 3.27908362e+00 -1.41877469e-04  1.14262767e+01 -8.16447685e+00
  -1.04999490e+01 -1.17084864e+01 -1.16118025e+01  1.74784539e+01]
 [-1.15886685e+01 -2.47276639e+01 -7.99538710e+00 -2.63404913e+00
   1.99669519e+00 -4.98084665e+00 -7.54816271e-01 -2.95445439e+01]
 [-1.20822131e+01  1.46441437e+01  2.56146204e+01 -2.59141373e+01
  -2.35770006e+00  4.36164149e+00 -2.91432508e+00 -1.01456918e+00]
 [-1.01044415e+00  7.83456302e-03 -1.64957372e+01  1.09388629e+01
  -7.23353132e+00  6.29042940e-01 -2.19868673e+00 -1.18607072e+01]
 [ 4.78482485e-01  5.88773458e-01  3.09294080e+00 -1.15621375e+01
  -1.86912222e+00 -1.40097370e+01 -1.14799391e+01  1.27173520e+01]]
The best


matweight=[[ 17.52840776,  15.17287966,  -3.36803196, -13.52139702,  12.82434282,
    1.77908684,   7.79677081,  21.27364315],
 [ -9.27409486,   4.11667832,   0.44799216, -10.13097644,  26.53334651,
  -10.32943725,  -9.50137782,  12.77250735],
 [ -6.14747663,  18.61528189,  12.30682136, -24.2607244,  -37.56226349,
   -2.53922494,  -4.44259917, -12.60282504],
 [  1.55920097,  20.38182851, -14.65379246, -11.00220405,  -9.20604766,
  -18.38426479,  13.81815195, -31.91711274],
 [  5.32860692,  -9.83658591,  19.12701698,  37.09987979,  10.5038393,
  -25.00040808, -12.40064444,  27.18197154],
 [ -9.33568958,  -7.07912103, -24.52121861,  26.11545583, -34.48158781,
  -10.79170571,  -4.81773755,  34.48190323]]

best 60 minutes human 1935
The best NN performance with four neurons is 864
speed=[[3, -4], [-5, 4], [1, 2], [3, 0], [1, 1], [5, -5], [-3, 0], [-2, 3], [-2, 4], [0, -2], [-4, -2], [0, -4], [0, 1], [-2, -2], [3, 2], [0, -3], [0, -1], [3, 5], [1, -5], [2, 5], [-1, -3], [4, 4], [2, 0], [-3, 1], [5, 4]]
detection=80


#Three neurons eulusoftmax



matweight=[[  3.82541289,  -0.91615453,  15.01991577, -15.79849793,  -6.253534,
  -12.85658343,   7.95297352],
 [ -3.39752777,  22.95549996,   3.78177323, -12.08068028,   3.41379568,
   -0.84374576, -33.68244375],
 [  4.14835297,   7.69352664,  17.39080592,  10.47021964, -13.63312496,
   -9.82371318,  17.18277283],
 [ 12.7826494,   11.56721962, -19.50346206,   1.83919545, -13.63778793,
  -17.30308803,  -9.53689592],
 [  7.97970289, -21.09745331, -13.5950587,    8.50981584,   8.12590946,
   22.25485562,  -1.75837199],
 [ 11.53824609,  12.19868371 , -0.66144914,  13.40997005,  33.70918401,
   13.33052536, -22.86591942]]


matweight=[[  2.81170383,  -1.08831759,  14.19762729, -15.62325548,  -7.04311562,
  -14.06285617,   8.46614362],
 [ -4.39660497,  21.416,   2.2837, -12.657,   3.227,
   -0.677, -32.38],
 [  2.233,   7.936,  16.619,  11.84, -12.806,
   -7.76,  18.298],
 [ 12.79,   12.144, -17.52,   3.047, -13.349,
  -17.35,  -8.71],
 [  6.465, -22.15, -15.05,    9.612,   8.615,
   22.181,  -1.422],
 [ 9.83,  13.123 , -1.122,  14.54,  33.60,
   11.32, -21.98]]

'''

#best1353
epsilon=0.04
epsilon2=.025

matweight=[[ 3.57970841e+00+epsilon, -5.22180230e-03-epsilon,  1.46722253e+01-epsilon, -1.38748415e+01-epsilon,
  -5.65596058e+00-epsilon, -1.37475428e+01,  1.04101271e+01],
 [-4.51020965e+00-epsilon,  2.19907985e+01,  4.73311501e+00, -1.44663738e+01,
   4.79250678e+00, -1.88984513e+00, -3.41546983e+01],
 [ 1.56706408e-01,  6.06701849e+00,  1.57791652e+01,  8.82514081e+00,
  -1.38335626e+01, -5.43752242e+00,  1.75429017e+01],
 [ 1.38158341e+01,  1.25558214e+01, -1.57149073e+01,  1.69403113e+00,
  -1.45154178e+01, -1.56861715e+01, -7.09385951e+00],
 [ 4.17339850e+00, -1.93338221e+01, -1.53945587e+01,  8.94950708e+00,
   9.72758512e+00,  2.05680924e+01, -4.26377257e-01],
 [ 1.12761032e+01,  1.14531257e+01,  4.25591085e-01+epsilon,  1.45677532e+01,
   3.33358155e+01,  1.22274711e+01, -2.13544794e+01]]
matinc=np.random.rand(numberofinput,numberofneurons)*deltamax-deltamax/2
matweight+=matinc

print (matweight)            
matweightold=matweight.copy()
add=np.zeros(s)
timegradientmatrix=np.zeros(s)
timegradientmatrixnew=np.zeros(s)
memi=0
memj=0
checktime=4 #3 for gradient ####tipology of time checking the number 2 it considers separated times, 4 for genetic###### 
geneticalgo='on'  #####on case the genetic algorithm is on#######
geneticmat=np.random.rand(numberofgeneticmatrix,numberofinput,numberofneurons)*maxsetmat-maxsetmat/2
###only in case of a matrix calculated before
for i in range(numberofgeneticmatrix):
    geneticmat[i]=(matweight.copy())
########
memgenetictime=[0 for i in range(numberofgeneticmatrix)]

print (geneticmat)
timeclock_series=[]
temp=-1
startgrad=[-1]
memtimelist=[]
memimprovedstep=[]
print (matweight)
key=input("Press a key:=")
for i in range(numberofinput):
    timeclock_series.append([0 for i in range(numberofneurons)])

    
print ("afd")    
print (add)    

print (matweight)    

ball_settings()
print (speed)
cgx[0]=-1
cgy[0]=0
contgeneticstep=-1
for v in range(nattemps):
    
    print ("start attempt")
    print (nattemps)
    if nattemps==0:
        break
    speed=[]
    ball_settings()
   # speed=[[2,-3],[-2,-1],[-1,2],[3,2],[2,-4],[-2,-2],[4,-1],[-3,-2],[4,-2],[-2,3]]
 #   speed=[[1,-1],[3,2],[-1,-3],[2,2],[1,1],[-1,3],[-4,-2],[3,2],[4,-1],[-2,-1]]
    #speed=[[1,-2],[3,2],[-1,-3],[2,2],[1,1],[-1,3],[-4,-2],[3,2],[4,-1],[-2,-1],[1,-1],[3,2],[-1,-3],[2,2],[1,1],[-1,3],[-4,-2],[3,2],[4,-1],[-2,-1]]
    #speed=[[1,-2],[3,2],[-1,-3],[2,2],[1,1],[-1,3],[-4,-2],[3,2],[4,-1],[-2,-1],[1,-1],[3,2],[-1,-3],[2,2],[1,1],[-1,3],[-4,-2],[3,2],[4,-1],[-2,-1],[1,-2],[3,2],[-1,-3],[2,2],[1,1],[-1,3],[-4,-2],[3,2],[3,-2],[2,1],[1,-1],[-3,2],[-3,-4],[-2,-2],[1,3],[1,3],[4,2],[-3,2],[3,-1],[-2,-3]]
#########   It was considered for the report##############
    speed=[[3, -4], [-5, 4], [1, 2], [3, 0], [1, 1], [5, -5], [-3, 0], [-2, 3], [-2, 4], [0, -2], [-4, -2], [0, -4], [0, 1], [-2, -2], [3, 2], [0, -3], [0, -1], [3, 5], [1, -5], [2, 5], [-1, -3], [4, 4], [2, 0], [-3, 1], [5, 4]]
####################################################
    contgeneticstep+=1
    temp+=1
    ball=[]
    ballrect=[]

    tlist=[]
    poslist=[]
    keylist=[]
    keystep=[]
    if v%matrixtimechanging==0 and v!=0:
        matweight=np.random.rand(numberofinput,numberofneurons)*maxsetmat-maxsetmat/2
        matweightold=matweight.copy()
        
    if geneticalgo=='on':
        print (geneticmat[contgeneticstep])
        matweight=geneticmat[contgeneticstep].copy()
        print (deltamin,deltamax)
    print ("matweight on")
    print (matweight)   
    if v%restart==0 and v!=0:
        deltamin=memdeltamin
        deltamax=memdeltamax
    if v%nreductionlim==0 and v!=0:
        deltamax/=2.0
        deltamin/=2.0   
    for i in range(nballs):
        ball.append(pygame.image.load("ball.png"))
        ballrect.append( ball[i].get_rect())
        ballrect[i].y=i*2
        ballrect[i].x=i*2

    if v>starttinyincrement:
        deltamax=tinydelta
        deltamin=-tinydelta
        standardev=tinydelta
        print ("tinydelta",tinydelta)
    player=pygame.image.load("rect.png")
    playerrect=player.get_rect()
    playerrect.x=width/2
    playerrect.y=height/2

    txt='s'

    for i in range(numberofinput):
        for j in range(numberofneurons):
            if typeofincrement=='determonedirection':
                pass
            
            if typeofincrement=='doubledirection':
                if v<switchuniformtogaussian:
                    if v%2==0 and v>1:
                        
                        add[i][j]=np.random.uniform(deltamin,deltamax)
                        
                       # print (add)
                        print ("action add")
                    elif v%2!=0 and v>1:
                        print("signadd")
                        add[i][j]=-add[i][j]
                        #print (matweightold)
                        
                        #print(add)

                else:
                    if (v%2==0 and v>1):
                        
                        add[i][j]=np.random.normal(0,standardev)
                        print ("gaussian",add[i][j])
                        
                       # print ("action add")
                    elif v%2!=0 and v>1:
                        #print("signadd")
                        add[i][j]=-add[i][j]

            if typeofincrement=='fourthdirection':

                pass

######here if you don't use the genetic algorithm you have to set another method below######
    if temp!=0:        
      #  gradient_method(deltagrad,numberofneurons,numberofinput)
        pass
      
     
            


    #matweight=total_variational_method(matweight)
    #print (matweight)
    #print(matweightold)

   # single_ordered_variational_method(memi,memj)
   # single_variational_method(numberofinput,numberofneurons)
    #product_variational_method()

        
            

    timeclock=0
    numberofmovement=0
    manualstart=1   
    while txt=='s':
        
        keys=pygame.key.get_pressed()
        if keys[K_q]:
            nattemps=0
            print ("fine")
            
        drawtheborder(limplayx,limplayy)

        timeclock+=1
        tlist=[]
        poslist=[]
        keylist=[]
        keystep=[]

        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        for i in range(nballs):
            #print "x"
            ballrect[i] = ballrect[i].move(speed[i])
            ballmoving(i) 
            screen.blit(ball[i], ballrect[i])
        
        
        pygame.display.flip() 
        screen.fill(white)
        screen.blit(player,playerrect)

        
        for i in range(nballs):
            if ballrect[i].x+ballrect[i].width/2<playerrect.x+playerrect.width and ballrect[i].x+ballrect[i].width/2>playerrect.x and ballrect[i].y+ballrect[i].height/2<playerrect.y+playerrect.height and ballrect[i].y+ballrect[i].height/2>playerrect.y:
                print ("process")
                print (timeclock,timeclockold)

                if checktime==4:   ####it's for genetic algortithm######
                    memgenetictime[contgeneticstep]=timeclock
                    if contgeneticstep==numberofgeneticmatrix-1:
                        geneticmat=genetic_algorithm(geneticmat,numberofgeneticmatrix,numberofaceptedgeneticmat,numberofinput,numberofneurons,deltamax,deltamin).copy()
                        contgeneticstep=-1
                if checktime==3:               ####it's for the gradient method#####
                    if timeclock>=timeclockold:
                        #matweightold=matweight.copy()
                        timeclockold=timeclock
                        memimprovedstep.append(timeclock)
                if checktime==1:              #####it's for total variatonal method, and single_variational method######
                    if timeclock>=timeclockold:
                        print ('TIMECLOCK',timeclock)
                        matweightold=matweight.copy()
                        timeclockold=timeclock
                        memimprovedstep.append(timeclock)
                        
                        

                    else:
                        for i in range(numberofinput):
                            for j in range(numberofneurons):
                                
                                matweight[i][j]=matweightold[i][j]                            
                    print (matweightold) 
                if checktime==2:   ####it's for single ordered variational method################
                    if timeclock>=timeclock_series[memi][memj]:
                        timeclock_series[memi][memj]=timeclock
                        matweightold[memi][memj]=matweight[memi][memj]
                        memimprovedstep.append(timeclock)
                    else:
                        
                        print (matweightold)
                        for i in range(numberofinput):
                            for j in range(numberofneurons):
                                
                                matweight[i][j]=matweightold[i][j]
                    print (timeclock_series)            
                    print (matweightold)                
                memj+=1
                if memj==numberofneurons:
                    memj=0
                    memi+=1
                    if memi==numberofinput:
                        memi=0                   

                    
                txt='f'       
        #processo attivazione nueroni e movimento con keylist


        for k in range(nballs):

            dx=-(playerrect.x+playerrect.width/2)+ballrect[k].x
            dy=-(playerrect.y+playerrect.height/2)+ballrect[k].y            

            vball=np.sqrt(speed[k][0]**2+speed[k][1]**2)
            vballx=speed[k][0]
            vbally=speed[k][1]

            dvec=np.array([dx,dy])
            dvec=dvec/np.linalg.norm(dvec)

            velvec=np.array([speed[k][0],speed[k][1]])
            velvec=velvec/np.linalg.norm(velvec)

            vdotv=np.dot(dvec,velvec)
            detection=80 #the cut off when the detection of ball start for the variable updx,dwdx,upsx,dwsx and vballx,vbally,vdotv
            if np.sqrt(dx**2+dy**2)>detection:
                vdotv=0
                vballx=0
                vbally=0
            invdx=0
            invdy=0
            updx=0
            dwsx=0
            upsx=0
            dwdx=0
            dsidexdx=0
            dsideydw=0
            dsidexsx=0
            dsideyup=0
            
            
            #if dx>0 and dx<(playerrect.width/2+detection):
             #   invdx=1
                
            #if dx<0 and dx>(-playerrect.width/2-detection):
             #   invdx=-1
                
            #if dy>0 and dy<(playerrect.height/2+detection):
             #   invdy=1
                

            #if dy<0 and dy>(-playerrect.height/2-detection):
             #   invdy=-1
            if np.sqrt(dx**2+dy**2)<detection:
                invdx=1
                #print("activated")
            kset=100.0
            if np.sqrt(dx**2+dy**2)<detection:
                if dx>0 and dy>0:
                    dwdx=kset/np.sqrt(dx**2+dy**2)
                if dx>0 and dy<0:
                    updx=kset/np.sqrt(dx**2+dy**2)
                if dx<0 and dy>0:
                    dwsx=kset/np.sqrt(dx**2+dy**2)
                if dx<0 and dy<0:
                    upsx=kset/np.sqrt(dx**2+dy**2)
            #kset=0.1      
            dsidexsx=kset/(playerrect.x+playerrect.width/2-(width/2-limplayx))
            dsideyup=kset/(playerrect.y+playerrect.height/2-(height/2-limplayy))
            dsidexdx=kset/(width/2+limplayx-(playerrect.x+playerrect.width/2))
            dsideydw=kset/(height/2+limplayy-(playerrect.y+playerrect.height/2))
            kset=0.01
            centerdist=kset*np.sqrt((playerrect.x-width/2)**2+(playerrect.y-height/2)**2)
            vtimesvdotv=np.sqrt(vballx**2+vbally**2)*vdotv            
            
#####here you choose which neural network or classic algorithm to use############            
        
            #print (vdotv)      
           # tipo='sigmoide'
            #tipo='random'
            #tipo='heaviside'
           # tipo='manual'
            #tipo="eluandsigmo"
            #tipo='linesigmosoft'
           # tipo='doubleheaviside'
            #tipo='linesoft'
           # tipo='elusoft'
            tipo='elusoftmax'
            #tipo='lineheaviside'
          #  tipo="eluheaviside"
           # tipo='elusoftlim'

            if tipo=='manual':
                if manualstart==1:
                    pygame.mouse.set_pos([width/2,height/2])
                    #print('got mouse')
                    #print (pygame.mouse.get_pos()[0],pygame.mouse.get_pos()[1])
                playerrect.x=pygame.mouse.get_pos()[0]
                playerrect.y=pygame.mouse.get_pos()[1]

                if pygame.mouse.get_pos()[0]!= width/2 or pygame.mouse.get_pos()[1]!= height/2:      
                    manualstart=0
                #print (pygame.mouse.get_pos())
                if playerrect.x < width/2-limplayx:
                    playerrect.x=width/2-limplayx
                if playerrect.x+playerrect.width > width/2+limplayx:
                    playerrect.x=width/2+limplayx-playerrect.width
                if playerrect.y < height/2-limplayy:
                    playerrect.y=height/2-limplayy
                if playerrect.y+playerrect.height > height/2+limplayy:
                    playerrect.y=height/2+limplayy-playerrect.height                

            for el in neuron(matweight,numberofneurons,numberofinput,invdx,invdy,vball,vdotv,tipo,vballx,vbally,updx,dwdx,upsx,dwsx,dsidexdx,dsideyup,dsidexsx,dsideydw,centerdist,vtimesvdotv,numberofneurons_layer2,numbinputfirst,numbinputsec,numbneuronsfirst,numbneuronsecond):

                if el==0:
                    playerrect.x-=1
                    numberofmovement+=1
                    
                if el==1:               
                
                    playerrect.y-=1
                    numberofmovement+=1
                if el==2:               
                    playerrect.x+=1
                    numberofmovement+=1
                    
                if el==3:               
                    
                    playerrect.y+=1
                    numberofmovement+=1


                    
                if playerrect.x < width/2-limplayx:
                    playerrect.x=width/2-limplayx
                if playerrect.x+playerrect.width > width/2+limplayx:
                    playerrect.x=width/2+limplayx-playerrect.width
                if playerrect.y < height/2-limplayy:
                    playerrect.y=height/2-limplayy
                if playerrect.y+playerrect.height > height/2+limplayy:
                    playerrect.y=height/2+limplayy-playerrect.height                


        red = (255,0,0)
        
        clock.tick(60)
    ##############gradient calculation############
    if temp!=0:
        #print (startgrad)
        if startgrad[0]==1:
            timegradientmatrixnew[cgy[0]][cgx[0]]=np.abs(1.0/timeclock)
            
            matweight[cgy[0]][cgx[0]]-=deltagrad
            print ("start",cgy[0],cgx[0])
            print (timegradientmatrix)
            print (timegradientmatrixnew)
        if startgrad[0]==0:
            for u in range(numberofinput):
                for w in range(numberofneurons):
                    matweight[u][w]-=gradcoef*(timegradientmatrixnew[u][w]-timegradientmatrix[u][w])/deltagrad
                   # print (gradcoef*(timegradientmatrixnew[u][w]-timegradientmatrix[u][w])/deltagrad)
            print (matweight)    
            cgx[0]=-1
            cgy[0]=0
            temp=-1

    if temp==0:
        for u in range(numberofinput):
            for w in range(numberofneurons):

                timeclockgrad=timeclockold
                if v==0:
                    timegradientmatrix[u][w]=np.abs(1.0/timeclock)
                if v!=0:
                    timegradientmatrix[u][w]=timegradientmatrixnew[u][w]
                    
        cgx[0]=-1
        cgy[0]=0
    #end of gradient###########################
    if checktime!=3:    
        memtimelist.append(timeclock)
    if checktime==3 and temp==0:
        memtimelist.append(timeclock)
        
graphquestion=input("Do I have to save the graph?=")

if graphquestion=="Y" or graphquestion=="y":
    namefile=input("Insert name of the file=")
    plt.plot(memtimelist)    
    plt.savefig("files/"+str(namefile))
    plt.show()
    plt.plot(memimprovedstep)
    plt.show()
    f1=open("files/"+str(namefile)+str('.dat'),"w")
    for el in memtimelist:
        f1.write(str(el)+str('\n'))
        
    f1.close()      
    f1=open("files/"+str(namefile)+str('.stp'),"w")

    for el in memimprovedstep:
        f1.write(str(el)+str('\n'))
    f1.close() 
print (matweight)
print(matweightold)
print (np.exp(valsoftb)/sumsoftb)
print (sumsoftb)
