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

def neuron(mat,numberofneurons,numberofinput,dx,dy,vball,vdotv,tipo,vballx,vbally,updx,dwdx,upsx,dwsx,vtimesvdotv,numberofneurons_layer2,numbinputfirst,numbinputsec,numbneuronsfirst,numbneuronsecond):

#    v=[vballx,vbally,vdotv,updx,dwdx,upsx,dwsx]
#    v=[vtimesvdotv,vdotv,updx,dwdx,upsx,dwsx]
    v=[vballx,vbally,updx,dwdx,upsx,dwsx]
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
        var=0.6    
        for k in range(numbneuronsfirst,numberofneurons):
            valneur=np.exp(valsofte[k-numbneuronsfirst])/sumsofte
            #var=np.random.uniform()
            if valneur>var:
                var=valneur
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
    
def total_variational_method():     
    for i in range(numberofinput):
        for j in range(numberofneurons):
            matweight[i][j]=matweightold[i][j]+add[i][j]                          
    print("AdD")
    print(add)
    print (deltamin,deltamax)
def single_variational_method(numberofinput,numberofneurons):     
    vc=np.random.uniform(deltamin,deltamax)
    i=np.random.randint(0,numberofinput)
    j=np.random.randint(0,numberofneurons)
    matweight[i][j]=matweightold[i][j]+vc                            

def single_ordered_variational_method(memi,memj):
    vc=np.random.uniform(deltamin,deltamax)
    matweight[memi][memj]=matweightold[memi][memj]+vc                            
    
def product_variational_method():
    vc=np.random.uniform(deltamin,deltamax)
    #print (vc)
    for i in range(numberofinput):
        for j in range(numberofneurons):
            matweight[i][j]=matweightold[i][j]*vc 
    
 
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

nattemps=2 #######nattemps number of attemps you want to do########
nreductionlim=100 #######any nreductionlimit attempted the random decreasing become half########
starttinyincrement=2001 ######after this limit it start with a tiny incrememt#######
timegradientcomparing=0
matrixtimechanging=2001
maxsetmat=4
restart=2002
switchuniformtogaussian=2001  ######after this number steps start the gaussian increment in case of total_method_variation#####
standardev=0.125     #####deviation standard for the gaussian increment############
numberofgeneticmatrix=12
numberofaceptedgeneticmat=6
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

typeofincrement=''
weightold=[]
cgx=[0]
cgy=[0]
add=[]
np.random.seed() 
limplayx=150      
limplayy=120
deltamin=-0.#######minimal increment during the back propagation########
deltamax=0.#######maximal increment during the back propagation########
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
#matweight=np.array([[ 0.22094751,  0.94078707, -0.02072097,  0.69578711],
# [ 0.44947971, -0.02383631,  0.61632851,  0.398313  ],
# [ 0.69090793,  0.89430503,  0.88635832,  0.54673588],
# [ 0.56736406,  0.35487388,  0.34985211,  0.90901892],
# [ 0.53553829, -0.01809768,  0.29734252,  0.88692097],
# [ 0.13315029,  1.17804209,  0.15164971,  0.58891307],
# [ 0.28070917,  0.0689819 ,  0.24363284,  0.52514285]])

#matweight=np.array([[14.19276602, -5.89938674, -9.0141398,2.56627123],
# [ 1.42963426,  4.23811138, -1.75191401,  2.27903732],
# [ 4.37504691, -7.23808026, -3.23301731,  0.1571672 ],
# [13.99738669, -7.75171274, -4.07108129, -1.71076681],
# [ 4.77392355, -0.14908269,  1.01437524,  6.54605288],
# [ 1.96141074,  1.42206272,  0.81261641, -5.73159898]])
#######exper4elusoft max time 449########################################
#[[ -74.24348901 -100.82620671  -36.97136553 -149.12248361  -65.57035022
#   -55.04717115  -51.57326793   92.71344571 -108.56378459  110.37402148]
# [  -3.31357933    5.70673574  -20.55525464  -69.55659148  -61.74172774
#   -64.82654369  -35.4919808   -20.54468867   45.59128135  162.25208416]
# [  -9.35219895  -15.71277787  -20.26932881  106.77836434  106.65702382
#  -102.13821136  -27.33461058  119.58168785  -23.47332645   53.4004991 ]
# [ 174.07120293   49.94401646   54.49830688   -2.34334759 -115.01679808
#    98.32794774  -63.84157156   21.65740228  -36.78310233 -165.45901731]
# [ -23.23337017   59.57838013   20.12587033  -92.80379912 -159.94806661
#    66.98131171   56.14157441   86.93323805    2.98228782   70.4648569 ]
# [  53.35418884 -121.82532408  -54.86888852   31.17005546   27.68148276
#    14.21024441   69.17941301  -43.22550418  143.50668802   55.93151842]]
#######################################################################

#######exper6elusoft max time 743########################################

#matweight=np.array([[ -55.46853696,  -74.74208045, -143.69391995,    2.98518223,  -62.56319491,
#   -27.06208611,  -88.58519295,   69.31096892,  -79.97831633,   93.71041058],
# [  13.8075652,    41.78890896,   19.6781344,   -73.09588311,   57.98069223,
#    -8.11959264,  -69.82385054, -117.75073543,  -47.18349613,   47.11398848],
# [ -83.2569441,    72.7863998,     7.95570047,   10.05858242,   88.0906438,
#    -8.30346416,  -39.62409031,   38.22155552,  -34.55579875,   34.59008892],
# [ -38.79683946,   70.31659394,   38.26965279,   -2.47204604,   80.82729193,
#   -32.27697094,  -31.93358023,   44.63651087,   87.30542335,   63.56094601],
# [  12.30355297,   99.36465528,   -5.32812858,   22.55671578,   54.62116471,
#    37.40641627,  -85.27406036,  -38.41017117,  -67.12817878,  -45.17835064],
# [ -58.02890664,  121.8480771,    80.44127287,   78.90172619,  -37.8209516,
#   -93.58602076,   22.65010987,  -42.15217114,    1.94769844,  -28.51224262]])
############################################################################
#####experheav6###################################################
#matweight=np.array([[ 63.34318438,  61.30619293, -43.79431469, -10.58421613],
# [ 34.42686682, -30.34561652,  37.12561437,  98.09239694],
# [-39.56505468, -12.70460137, -19.28512674,   2.39608693],
# [ 45.75514013,  24.67024589, -27.18256676, -24.70445515],
# [-10.88793694,  46.03208127, -76.69253545, -27.04133687],
# [ 38.70715376,  42.72399733,  61.42855631,  -8.78776631]])
#########################################################################
###########################
#matweight=np.array([[-2.04130287,  7.53250231,  6.97256607,  3.42596826,  3.41298108, -2.4840838,
#  -0.89136838,  3.97305206, -4.2590926,  -1.28420292],
# [-2.93011187, -1.98500066, -2.18333482,  1.61332449, -3.29507038, -0.93122476,
#   6.56573907,  1.20171406, -3.04848612,  3.29408945],
# [-3.91721742,  0.24219721, -1.81845288, -3.31881178,  3.28802832, -6.75426192,
#   3.04896353, -6.9568943,  -2.69687904, -0.14820766],
# [ 2.01138085, -4.84240229, -1.63177378, -5.27926453,  0.6400804,   0.19576328,
#   0.45183488, -5.63743533,  0.20214788, -3.43634634],
# [-2.68432734, -3.18286695, -0.39476294, -1.90874889,  3.0980342,  -0.11297722,
#   2.466131,   -7.11823555,  2.62899575, -1.17758005],
# [-2.56563575,  5.19969988, -1.56468926, -1.85187876, -3.86502798, -5.36028287,
#   3.46365583,  2.2983075,   3.30964542, -7.68952196]])

###############################################
matweight=np.array([[-32.01756724,  -8.78998167, -49.93030866, -23.46873766, -61.70305418,
  -39.0642101,   -4.53581764, -39.06559178,  48.96842672],
 [ 59.39449323, -44.10770515,  24.80436888,  24.78353701,  -8.47882157,
   10.32072324,  63.09455426,  -4.84401403,  -0.65597992],
 [-58.36615066,  18.52903402,  27.26429688, -19.95604987,  18.15458268,
  -19.39641881,  58.63793997,  -5.30915539,  -1.62837691],
 [-48.75770943, -75.26611334, -20.27226638,   7.1941845,  -16.38561025,
  -38.75452752,  19.68997363, -36.70118816,  13.14409695],
 [-35.98782358, -17.81758673, -21.64858094,  13.62009275, -50.74859357,
  -66.53694376, -87.16245432,  22.31100201,  10.94439215],
 [-39.31672438, -37.53395044, -14.26305811, -26.18555246,  32.22210295,
  -87.74633532,  26.46389591, -75.96744418,   5.26420771]])

matweightold=matweight.copy()
add=np.zeros(s)
timegradientmatrix=np.zeros(s)
timegradientmatrixnew=np.zeros(s)
memi=0
memj=0
checktime=4 #3 for gradient ####tipology of time checking the number 2 it considers separated times, 4 for genetic###### 
geneticalgo='off'  #####on case the genetic algorithm is on#######
geneticmat=np.random.rand(numberofgeneticmatrix,numberofinput,numberofneurons)*maxsetmat-maxsetmat/2
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
                    if v%2==0:
                        
                        add[i][j]=np.random.uniform(deltamin,deltamax)
                        
                        
                        print ("action add")
                    else:
                        print("signadd")
                        add[i][j]=-add[i][j]

                else:
                    if v%2==0:
                        
                        add[i][j]=np.random.normal(0,standardev)
                        print ("gaussian",add[i][j])
                        
                       # print ("action add")
                    else:
                        #print("signadd")
                        add[i][j]=-add[i][j]

            if typeofincrement=='fourthdirection':

                pass

######here if you don't use the genetic algorithm you have to set another method below######
    if temp!=0:        
      #  gradient_method(deltagrad,numberofneurons,numberofinput)
        pass
      
     
            


   # total_variational_method()

   # single_ordered_variational_method(memi,memj)
   # single_variational_method(numberofinput,numberofneurons)
    #product_variational_method()

        
            

    timeclock=0
    numberofmovement=0
   
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
                print("activated")
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
            #tipo='elusoftmax'
           # tipo='lineheaviside'
          #  tipo="eluheaviside"
            tipo='elusoftlim'  
            if tipo=='manual':
                playerrect.x=pygame.mouse.get_pos()[0]
                playerrect.y=pygame.mouse.get_pos()[1]
                #print (pygame.mouse.get_pos())
                if playerrect.x < width/2-limplayx:
                    playerrect.x=width/2-limplayx
                if playerrect.x+playerrect.width > width/2+limplayx:
                    playerrect.x=width/2+limplayx-playerrect.width
                if playerrect.y < height/2-limplayy:
                    playerrect.y=height/2-limplayy
                if playerrect.y+playerrect.height > height/2+limplayy:
                    playerrect.y=height/2+limplayy-playerrect.height                

            for el in neuron(matweight,numberofneurons,numberofinput,invdx,invdy,vball,vdotv,tipo,vballx,vbally,updx,dwdx,upsx,dwsx,vtimesvdotv,numberofneurons_layer2,numbinputfirst,numbinputsec,numbneuronsfirst,numbneuronsecond):

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
print (np.exp(valsoftb)/sumsoftb)
print (sumsoftb)
