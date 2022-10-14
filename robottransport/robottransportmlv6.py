# Robot transport ML #
import numpy as np
import pygame
import sys
import time


class robot:

    def __init__(self):
        self.x=0
        self.y=0
        self.orientation=""
        self.basex1=0
        self.basey1=0
        self.basex2=0
        self.basey2=0
        self.counter=0
        self.back=0
        self.move=0
        self.networktype=''
        self.angle=0
        self.listmov=[]
        self.nnphysicaltype=''
        self.parsigmo=0
        self.countmoving=1
        


    
    def random_moving(self,limplayx,limplayy):
        choice=np.random.randint(1,5)
        if choice==1:
            if self.orientation=='est':
                if self.x<limplayx-1:
                    self.x+=1
                
            else:
                self.orientation='est'
        if choice==2:
            if self.orientation=='ovest':
                if self.x>0:

                    self.x-=1
            else:
                self.orientation='ovest'
        if choice==3:
            if self.orientation=='sud':
                if self.y<limplayy-1:

                    self.y+=1
            else:
                self.orientation='sud'
        if choice==4:
            if self.orientation=='nord':
                if self.y>0:

                    self.y-=1
            else:
                self.orientation='nord'
            
                
        
        if (self.x==self.basex2 and self.y==self.basey2 and self.back==0):
            self.counter+=1
            self.back=1            
    
        if (self.x==self.basex1 and self.y==self.basey1 and self.back==1):
            self.counter+=1
            self.back=0            

    def moving(self):
        self.countmoving+=1  
        if self.listmov[0]==1:
            self.angle+=90
            if self.angle==360:
                self.angle=0
 
            if self.angle==0:
                self.orientation='est'
            if self.angle==90:
                self.orientation='nord'
            if self.angle==180:
                self.orientation='ovest'
            if self.angle==270:
                self.orientation='sud'
              
        if self.listmov[1]==1:
            if self.orientation=='est':
                if self.x<limplayx-1:
                    self.x+=1
                
            if self.orientation=='ovest':
                if self.x>0:
                    self.x-=1
                
            if self.orientation=='sud':
                if self.y<limplayy-1:
                    self.y+=1

            if self.orientation=='nord':
                if self.y>0:
                    self.y-=1
            self.countmoving=1        
        if (self.x==self.basex2 and self.y==self.basey2 and self.back==0):
            self.counter+=1
            self.back=1            
    
        if (self.x==self.basex1 and self.y==self.basey1 and self.back==1):
            self.counter+=1
            self.back=0                            
                
        
    def neural_network(self,n1,n2,n3,n4,n5):
        inputs=[0 for i in range(n1)]
        #inputs[6]=0.1
        inputs[7]=5*(1-np.sqrt(1/self.countmoving))

        for agent in robots:
            if self.nnphysicaltype=='nn1':
                if self.orientation=='est':
                    if (agent.x==self.x+1 and agent.y==self.y):
                        inputs[0]=1
                    if (agent.x==self.x+1 and agent.y==self.y+1):
                        inputs[1]=1
                    if (agent.x==self.x+1 and agent.y==self.y-1):
                        inputs[2]=1
                    if (agent.x==self.x+2 and agent.y==self.y):
                        inputs[3]=1
                    if (agent.x==self.x+2 and agent.y==self.y+1):
                        inputs[4]=1
                    if (agent.x==self.x+2 and agent.y==self.y-1):
                        inputs[5]=1
                if self.orientation=='ovest':
                    if (agent.x==self.x-1 and agent.y==self.y):
                        inputs[0]=1
                    if (agent.x==self.x-1 and agent.y==self.y+1):
                        inputs[2]=1
                    if (agent.x==self.x-1 and agent.y==self.y-1):
                        inputs[1]=1
                    if (agent.x==self.x-2 and agent.y==self.y):
                        inputs[3]=1
                    if (agent.x==self.x-2 and agent.y==self.y+1):
                        inputs[5]=1
                    if (agent.x==self.x-2 and agent.y==self.y-1):
                        inputs[4]=1

                if self.orientation=='nord':
                    if (agent.x==self.x and agent.y==self.y-1):
                        inputs[0]=1
                    if (agent.x==self.x+1 and agent.y==self.y-1):
                        inputs[1]=1
                    if (agent.x==self.x-1 and agent.y==self.y-1):
                        inputs[2]=1
                    if (agent.x==self.x and agent.y==self.y-2):
                        inputs[3]=1
                    if (agent.x==self.x-1 and agent.y==self.y-2):
                        inputs[5]=1
                    if (agent.x==self.x+1 and agent.y==self.y-2):
                        inputs[4]=1

                if self.orientation=='sud':
                    if (agent.x==self.x and agent.y==self.y+1):
                        inputs[0]=1
                    if (agent.x==self.x+1 and agent.y==self.y+1):
                        inputs[2]=1
                    if (agent.x==self.x-1 and agent.y==self.y+1):
                        inputs[1]=1
                    if (agent.x==self.x and agent.y==self.y+2):
                        inputs[3]=1
                    if (agent.x==self.x-1 and agent.y==self.y+2):
                        inputs[4]=1
                    if (agent.x==self.x+1 and agent.y==self.y+2):
                        inputs[5]=1


                if self.back==0:
                    if self.orientation=='est' and self.basex2>self.x:
                        inputs[6]=1
                    if self.orientation=='ovest' and self.basex2<self.x:
                        inputs[6]=1
                    if self.orientation=='nord' and self.basey2<self.y:
                        inputs[6]=1
                    if self.orientation=='sud' and self.basey2>self.y:
                        inputs[6]=1

                if self.back==1:
                    if self.orientation=='est' and self.basex1>self.x:
                        inputs[6]=1
                    if self.orientation=='ovest' and self.basex1<self.x:
                        inputs[6]=1
                    if self.orientation=='nord' and self.basey1<self.y:
                        inputs[6]=1
                    if self.orientation=='sud' and self.basey1>self.y:
                        inputs[6]=1
        
  

        sumnorm=0
        for i in range(len(inputs)):
            sumnorm+=inputs[i]**2
            
        sumnorm=np.sqrt(sumnorm)
        if sumnorm!=0:
            inputs=inputs/sumnorm
        layer1=[]
        if self.networktype=='doubleheavsidelayer':
            sumv=0
            for i in range(n2):
                sumv=0
                for j in range(n1):
                    sumv+=mat[0][j][i]*inputs[j]
            
                if sumv>0:
                    layer1.append(1)
                else:
                    layer1.append(-1)
            output=[]        
            for i in range(n3):
                sumv=0
                for j in range(n2):
                    sumv+=mat[1][j][i]*layer1[j]
            
                if sumv>0:
                    output.append(1)
                else:
                    output.append(0)
            self.listmov=output.copy()
        if self.networktype=='heavsidesigmolayer':
            sumv=0
            for i in range(n2):
                sumv=0
                for j in range(n1):
                    sumv+=mat[0][j][i]*inputs[j]
            
                if sumv>0:
                    layer1.append(1)
                else:
                    layer1.append(-1)
            output=[]        
            for i in range(n3):
                sumv=0
                for j in range(n2):
                    sumv+=mat[1][j][i]*layer1[j]
                    
                ris=1/(np.exp(-self.parsigmo*sumv)+1)
                cas=np.random.uniform(0,1)
                if cas<ris:
                    output.append(1)
                else:
                    output.append(0)
            self.listmov=output.copy()            
##############controllare che la rete sia implementata giusta################                    
            print ("***********************")
#            print (output)
            
            
        
        
        
    
def draw_scenario(limplayx,limplayy,dl):
    pygame.draw.rect(screen,(0,0,255),(width/2-limplayx*dl/2.0,height/2-limplayy*dl/2.0,limplayx*dl,limplayy*dl),2)
    numbx=int(limplayx)
    numby=int(limplayy)
    for i in range(numbx-1):
        pygame.draw.line(screen,(0,0,255),(width/2-limplayx*dl/2.0+(i+1)*dl,height/2-limplayy/2.0*dl),(width/2-limplayx/2.0*dl+(i+1)*dl,height/2+limplayy/2.0*dl),1)
    for j in range(numby-1):
        pygame.draw.line(screen,(0,0,255),(width/2-limplayx*dl/2.0,height/2-limplayy*dl/2.0+(j+1)*dl),(width/2+limplayx*dl/2.0,height/2-limplayy*dl/2.0+(j+1)*dl),1)
        
        
        
    
def draw_robots(x,y,orientation,dl,rc,start,abx1,aby1,abx2,aby2):
    
    if start==1:
        colx=np.random.randint(50,255)
        coly=np.random.randint(50,255)
        colz=np.random.randint(50,255)
        color.append([colx,coly,colz])
        
    if orientation=='est':
        pygame.draw.polygon(screen,color[rc],([(width/2-limplayx/2*dl)+x*dl,(height/2-limplayy/2*dl)+y*dl],[(width/2-limplayx*dl/2)+x*dl+dl,(height/2-limplayy*dl/2)+y*dl+dl/2],[(width/2-limplayx*dl/2)+x*dl,(height/2-limplayy*dl/2)+y*dl+dl]))
        pygame.draw.circle(screen,color[rc],([int((width/2-limplayx*dl/2)+x*dl+dl),int((height/2-limplayy*dl/2)+y*dl+dl/2)]),2)
    if orientation=='ovest':
        pygame.draw.polygon(screen,color[rc],([(width/2-limplayx*dl/2)+x*dl+dl,(height/2-limplayy*dl/2)+y*dl],[(width/2-limplayx*dl/2)+x*dl,(height/2-limplayy*dl/2)+y*dl+dl/2],[(width/2-limplayx*dl/2)+x*dl+dl,(height/2-limplayy*dl/2)+y*dl+dl]))
        pygame.draw.circle(screen,color[rc],([int((width/2-limplayx*dl/2)+x*dl),int((height/2-limplayy*dl/2)+y*dl+dl/2)]),2)
    
    if orientation=='sud':
        pygame.draw.polygon(screen,color[rc],([(width/2-limplayx*dl/2)+x*dl,(height/2-limplayy*dl/2)+y*dl],[(width/2-limplayx*dl/2)+x*dl+dl/2,(height/2-limplayy*dl/2)+y*dl+dl],[(width/2-limplayx*dl/2)+x*dl+dl,(height/2-limplayy*dl/2)+y*dl]))
        pygame.draw.circle(screen,color[rc],([int((width/2-limplayx*dl/2)+x*dl+dl/2),int((height/2-limplayy*dl/2)+y*dl+dl)]),2)

    if orientation=='nord':
        pygame.draw.polygon(screen,color[rc],([(width/2-limplayx*dl/2)+x*dl,(height/2-limplayy*dl/2)+y*dl+dl],[(width/2-limplayx*dl/2)+x*dl+dl/2,(height/2-limplayy*dl/2)+y*dl],[(width/2-limplayx*dl/2)+x*dl+dl,(height/2-limplayy*dl/2)+y*dl+dl]))
        pygame.draw.circle(screen,color[rc],([int((width/2-limplayx*dl/2)+x*dl+dl/2),int((height/2-limplayy*dl/2)+y*dl)]),2)
      #  pygame.draw.circle(screen,color[rc],([100,100]),10)

    
    pygame.draw.rect(screen,color[rc],([(width/2-limplayx*dl/2)+abx1*dl,(height/2-limplayy*dl/2)+aby1*dl],[dl,dl]))
    pygame.draw.rect(screen,color[rc],([(width/2-limplayx*dl/2)+abx2*dl,(height/2-limplayy*dl/2)+aby2*dl],[dl,dl]))



def single_increasingmat(n1,n2,n3,n4,n5,n6,incmax):
    matc=np.random.rand(n1,n2)*incmax-incmax/2
    mat[0]=memmat[0]+matc
    matc=np.random.rand(n3,n4)*incmax-incmax/2
    mat[1]=memmat[1]+matc

    matc=np.random.rand(n5,n6)*incmax-incmax/2
    mat[2]=memmat[2]+matc

    

pygame.init()
size = width, height = 600, 600
speed=[]
screen = pygame.display.set_mode(size)
background = pygame.image.load('Immagine.png').convert()
screen.blit(background, (0, 0))
white = 255, 255, 255
limplayx=20
limplayy=20
dl=20
robots=[]
nrobots=7
niterations=2500
nshots=2
maxsetmat=4
bestperformance=0
numberoflayers=2
numberofgenmatrix=10
numberofacceptedmat=10
reductionstep=3
mat=[[],[],[]]
memmat=mat.copy()
incmax=0.0
n1=8
n2=2
n3=n2
n4=2
n5=0
n6=0
mat[0]=np.random.rand(n1,n2)*maxsetmat-maxsetmat/2
mat[1]=np.random.rand(n3,n4)*maxsetmat-maxsetmat/2
mat[2]=np.random.rand(n5,n6)*maxsetmat-maxsetmat/2

#####matrice con pesi per doubleheavside con 175 di performance con 1000 steps nn1######

mat[0]=[[  6.80795718,  -2.06773695],
 [ -5.96595566,   4.5840838 ],
 [-12.03646219,  -8.11401996],
 [ -7.28067343,   2.92532532],
 [  5.55106764,  -8.70296109],
 [  0.08502666,  -1.17875238],
 [  1.41098879,   3.08336068],
 [  1.00735317,   5.88493413]]
mat[1]=[[-9.91842103,  5.63657717],
 [-1.39789372,  2.91120306]]
#################################
memmat[0]=mat[0].copy()
memmat[1]=mat[1].copy()
memmat[2]=mat[2].copy()

#for el in mat:
#    print (el)

#print (mat[1][1][0])    

#for i in range(numberoflayers):
#    matgen.append(mat[i])

#print (matgen)



for k in range(nrobots):
    
    robots.append(robot())
###### robot[0] #############
robots[0].x=1
robots[0].y=2
robots[0].orientation='nord'
#robots[0].nnphysicaltype='nn1'
robots[0].basex1=1
robots[0].basey1=2
robots[0].basex2=8
robots[0].basey2=4
#############################

#############################
robots[1].x=4
robots[1].y=7
robots[1].orientation='nord'
#robots[1].nnphysicaltype='nn1'

robots[1].basex1=4
robots[1].basey1=7
robots[1].basex2=6
robots[1].basey2=1
##############################

robots[2].x=8
robots[2].y=8
robots[2].orientation='nord'
#robots[2].nnphysicaltype='nn1'

robots[2].basex1=8
robots[2].basey1=8
robots[2].basex2=4
robots[2].basey2=4
#############################

robots[3].x=1
robots[3].y=9
robots[3].orientation='nord'
#robots[3].nnphysicaltype='nn1'

robots[3].basex1=1
robots[3].basey1=9
robots[3].basex2=10
robots[3].basey2=8
##############################

robots[4].x=10
robots[4].y=2
robots[4].orientation='nord'
#robots[4].nnphysicaltype='nn1'

robots[4].basex1=10
robots[4].basey1=2
robots[4].basex2=1
robots[4].basey2=10
#########################
robots[5].x=6
robots[5].y=4
robots[5].orientation='nord'
#robots[5].nnphysicaltype='nn1'

robots[5].basex1=6
robots[5].basey1=4
robots[5].basex2=10
robots[5].basey2=10


robots[6].x=6
robots[6].y=4
robots[6].orientation='nord'
#robots[5].nnphysicaltype='nn1'

robots[6].basex1=7
robots[6].basey1=2
robots[6].basex2=3
robots[6].basey2=10
for agent in robots:
    if agent.orientation=='est':
        agent.angle=0
    if agent.orientation=='ovest':
        agent.angle=180
    if agent.orientation=='nord':
        agent.angle=90
    if agent.orientation=='sud':
        agent.angle=270
        

color=[]
start=1

#robots[0].networktype='heavsidesinglelayer'
#robots[0].neural_network(n1,n2,n3,n4,n5)

filequestion=input('Do you want to save the data into the files y/n')
if filequestion=='y':
    f1=open('datarobxy.dat','w')
    f2=open('databasexy.dat','w')
    for k in robots:
        f2.write(str(k.basex1)+","+str(k.basey1)+","+str(k.basex2)+","+str(k.basey2))
        f2.write("\n")    
    
    f2.close()            
        
for h in range(nshots):
    #mat[0]=np.random.rand(n1,n2)*maxsetmat-maxsetmat/2
    #mat[1]=np.random.rand(n3,n4)*maxsetmat-maxsetmat/2
    #mat[2]=np.random.rand(n5,n6)*maxsetmat-maxsetmat/2 
    
    if h%reductionstep==0:
        incmax/=2.0
        
    prize=0
    #print(mat[0])
    #print(mat[1])
    for agent in robots:
        prize+=agent.counter
    if prize>=bestperformance:
        bestperformance=prize
        memmat[0]=mat[0].copy()
        memmat[1]=mat[1].copy()
        memmat[2]=mat[2].copy()
        
   # mat[0]=np.random.rand(n1,n2)*maxsetmat-maxsetmat/2
   # mat[1]=np.random.rand(n3,n4)*maxsetmat-maxsetmat/2
   # mat[2]=np.random.rand(n5,n6)*maxsetmat-maxsetmat/2

    ####single mat increasing subroutine need*********
    if h>0:
        single_increasingmat(n1,n2,n3,n4,n5,n6,incmax)
    print ("************new********************")
   # time.sleep(2)
    stop=0
    ############settare posizione iniziale robots######
    for agent in robots:
        agent.x=agent.basex1
        agent.y=agent.basey1
        agent.counter=0
        agent.back=0
        agent.listmov=[]
        agent.orientation='nord'
        agent.angle=90
        agent.countmoving=1
        

    for k in range(niterations):
        if stop==0:
            draw_scenario(limplayx,limplayy,dl)
            rc=0
            for agent in robots:
                draw_robots(agent.x,agent.y,agent.orientation,dl,rc,start,agent.basex1,agent.basey1,agent.basex2,agent.basey2)
                rc+=1
            rc=0
            nag=-1
            for agent in robots:
                nag+=1
                #agent.random_moving(limplayx,limplayy)
                if stop==0:
                    #agent.networktype='heavsidesigmolayer'
                    agent.networktype='doubleheavsidelayer'
                    agent.parsigmo=1.0
                    agent.nnphysicaltype='nn1'
                    agent.neural_network(n1,n2,n3,n4,n5)
                    #print (agent.listmov)
                    #print (agent.orientation)
                    agent.moving()
                    if filequestion=='y':
                        f1.write(str(nag)+","+str(agent.orientation)+","+str(agent.x)+","+str(agent.y))
                        f1.write("\n")    
                        

                    for n in range(len(robots)):
                        if n!=rc:
                            
                            if agent.x==robots[n].x and agent.y==robots[n].y:
                                stop=1
                                print ("crash")
                                #time.sleep(1000)
                                
                    rc+=1
            start=0
            for event in pygame.event.get():
                print (bestperformance)            
                #print (memmat[0])
                #print (memmat[1])

                if event.type == pygame.QUIT: sys.exit()
            pygame.display.flip() 
            screen.fill(white)
            time.sleep(0.1)
if filequestion=='y':
    f1.close()
    
print (bestperformance)            
print (memmat[0])
print (memmat[1])
print (incmax)

