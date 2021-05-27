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

def neuron(mat,numberofneurons,numberofinput,dx,dy,vball,vdotv,tipo,vballx,vbally,updx,dwdx,upsx,dwsx,dsidexdx,dsideyup,dsidexsx,dsideydw,vtimesvdotv,numberofneurons_layer2,numbinputfirst,numbinputsec,numbneuronsfirst,numbneuronsecond):

    v=[vballx,vbally,updx,dwdx,upsx,dwsx]
#    v=[vtimesvdotv,vdotv,updx,dwdx,upsx,dwsx]
   # v=[vballx,vbally,updx,dwdx,upsx,dwsx,dsidexdx,dsideyup,dsidexsx,dsideydw]
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
        var=0.8
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

nattemps=300 #######nattemps number of attemps you want to do########
nreductionlim=30 #######any nreductionlimit attempted the random decreasing become half########
starttinyincrement=2001 ######after this limit it start with a tiny incrememt#######
timegradientcomparing=0
matrixtimechanging=2001
maxsetmat=1
restart=2002
switchuniformtogaussian=2001  ######after this number steps start the gaussian increment in case of total_method_variation#####
standardev=0.125     #####deviation standard for the gaussian increment############
numberofgeneticmatrix=6
numberofaceptedgeneticmat=3
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
numberofneurons=5
numberofneurons_layer2=4 #it has to be 4!!!!
numberofinput=6

typeofincrement='doubledirection'
weightold=[]
cgx=[0]
cgy=[0]
add=[]
np.random.seed() 
limplayx=150      
limplayy=120
deltamin=-1#######minimal increment during the back propagation########
deltamax=1######maximal increment during the back propagation########
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


matweightold=matweight.copy()
add=np.zeros(s)
timegradientmatrix=np.zeros(s)
timegradientmatrixnew=np.zeros(s)
memi=0
memj=0
checktime=1 #3 for gradient ####tipology of time checking the number 2 it considers separated times, 4 for genetic###### 
geneticalgo='off'  #####on case the genetic algorithm is on#######
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
                        
                        print (add)
                        print ("action add")
                    elif v%2!=0 and v>1:
                        print("signadd")
                        add[i][j]=-add[i][j]
                        print (matweightold)
                        
                        print(add)

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
      
     
            


    matweight=total_variational_method(matweight)
    print (matweight)
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
                    
            dsidexsx=kset/(playerrect.x+playerrect.width/2-(width/2-limplayx))
            dsideyup=kset/(playerrect.y+playerrect.height/2-(height/2-limplayy))
            dsidexdx=kset/(width/2+limplayx-(playerrect.x+playerrect.width/2))
            dsideydw=kset/(height/2+limplayy-(playerrect.y+playerrect.height/2))
            
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
           # tipo='lineheaviside'
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

            for el in neuron(matweight,numberofneurons,numberofinput,invdx,invdy,vball,vdotv,tipo,vballx,vbally,updx,dwdx,upsx,dwsx,dsidexdx,dsideyup,dsidexsx,dsideydw,vtimesvdotv,numberofneurons_layer2,numbinputfirst,numbinputsec,numbneuronsfirst,numbneuronsecond):

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
