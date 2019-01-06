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

def neuron(mat,numberofneurons,numberofinput,dx,dy,vball,vdotv,tipo,vballx,vbally,updx,dwdx,upsx,dwsx,vtimesvdotv):

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
    geneticmat=matgennew.copy()
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

nattemps=800 #######nattemps number of attemps you want to do########
nreductionlim=100 #######any nreductionlimit attempted the random decreasing become half########
starttinyincrement=2001 ######after this limit it start with a tiny incrememt#######
timegradientcomparing=0
matrixtimechanging=2001
maxsetmat=1
restart=2002
switchuniformtogaussian=2001  ######after this number steps start the gaussian increment in case of total_method_variation#####
standardev=0.125     #####deviation standard for the gaussian increment############
numberofgeneticmatrix=15
numberofaceptedgeneticmat=5
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
numberofneurons=4
numberofinput=6
typeofincrement=''
weightold=[]
cgx=[0]
cgy=[0]
add=[]
np.random.seed() 
limplayx=120      
limplayy=100
deltamin=-4     #######minimal increment during the back propagation########
deltamax=4 #######maximal increment during the back propagation########
memdeltamin=deltamin
memdeltamax=deltamax
tinydelta=0.00125  #######the tinydelta increment when it start tinydelta modality######
deltagrad=0.25   #######delta variation in gradient method#########
awardingprize='false'
prizecoef=0.98
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

matweight=np.array([[14.19276602, -5.89938674, -9.0141398,2.56627123],
 [ 1.42963426,  4.23811138, -1.75191401,  2.27903732],
 [ 4.37504691, -7.23808026, -3.23301731,  0.1571672 ],
 [13.99738669, -7.75171274, -4.07108129, -1.71076681],
 [ 4.77392355, -0.14908269,  1.01437524,  6.54605288],
 [ 1.96141074,  1.42206272,  0.81261641, -5.73159898]])
matweightold=matweight.copy()
add=np.zeros(s)
timegradientmatrix=np.zeros(s)
timegradientmatrixnew=np.zeros(s)
memi=0
memj=0
checktime=4 #3 for gradient ####tipology of time checking the number 2 it considers separated times, 4 for genetic###### 
geneticalgo='on'  #####on case the genetic algorithm is on#######
geneticmat=np.random.rand(numberofgeneticmatrix,numberofinput,numberofneurons)*maxsetmat-maxsetmat/2
memgenetictime=[0 for i in range(numberofgeneticmatrix)]

print (geneticmat)
timeclock_series=[]
temp=-1
startgrad=[-1]
memtimelist=[]
memimprovedstep=[]
print (matweight)

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
    

  #  ball_settings()
#    speed=[[2,-3],[-2,-1],[-1,2],[3,2],[2,-4],[-2,-2],[4,-1],[-3,-2],[4,-2],[-2,3]]
 #   speed=[[1,-1],[3,2],[-1,-3],[2,2],[1,1],[-1,3],[-4,-2],[3,2],[4,-1],[-2,-1]]
   # speed=[[1,-2],[3,2],[-1,-3],[2,2],[1,1],[-1,3],[-4,-2],[3,2],[4,-1],[-2,-1],[1,-1],[3,2],[-1,-3],[2,2],[1,1],[-1,3],[-4,-2],[3,2],[4,-1],[-2,-1]]
   # speed=[[1,-2],[3,2],[-1,-3],[2,2],[1,1],[-1,3],[-4,-2],[3,2],[4,-1],[-2,-1],[1,-1],[3,2],[-1,-3],[2,2],[1,1],[-1,3],[-4,-2],[3,2],[4,-1],[-2,-1],[1,-2],[3,2],[-1,-3],[2,2],[1,1],[-1,3],[-4,-2],[3,2],[3,-2],[2,1],[1,-1],[-3,2],[-3,-4],[-2,-2],[1,3],[1,3],[4,2],[-3,2],[3,-1],[-2,-3]]
 #   speed=[[3, -4], [-5, 4], [1, 2], [3, 0], [1, 1], [5, -5], [-3, 0], [-2, 3], [-2, 4], [0, -2], [-4, -2], [0, -4], [0, 1], [-2, -2], [3, 2], [0, -3], [0, -1], [3, 5], [1, -5], [2, 5], [-1, -3], [4, 4], [2, 0], [-3, 1], [5, 4]]
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
      
     
            


  #  total_variational_method()

   # single_ordered_variational_method(memi,memj)
   # single_variational_method(numberofinput,numberofneurons)
    #product_variational_method()

        
            

    timeclock=0
    numberofmovement=0
   
    while txt=='s':
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
           # tipo='random'
            tipo='heaviside'
          #  tipo='manual'
             
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

            for el in neuron(matweight,numberofneurons,numberofinput,invdx,invdy,vball,vdotv,tipo,vballx,vbally,updx,dwdx,upsx,dwsx,vtimesvdotv):

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
