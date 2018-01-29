#dx->
#dy->   right or no
#up->
#dw->


import pygame
from pygame.locals import *
from sys import exit
import time
import random
pygame.init()
scenario=[]
size=width, height =600,400
#speed=[1,1]
black=0,0,
wx=15
wy=15
nwall=10
screen=pygame.display.set_mode(size)
background=pygame.image.load("ml3stuff/immagine.png").convert()
background = pygame.transform.scale(background, (width, height))
screen.blit(background,(0,0))
hero=pygame.image.load("ml3stuff/you.png").convert()
hero = pygame.transform.scale(hero, (wx,wy))
#screen.blit(hero,(wx,wy))
wall=[]
dw=1
up=0
dx=1000
dy=0
w=[random.uniform(0,0.25) for i in range(4)]
wold=[]
wold[:]=w[:]

def neuron(dx,dy,up,dw):
    print "neuron value" ,dx*w[0]+dy*w[1]+up*w[2]+dw*w[3]
    if dx*w[0]+dy*w[1]+up*w[2]+dw*w[3]>5:
        
        return "yes"
    else:
        return "no"
    
    
    
def check_crash(i,x):
    
    
    if x+wx>kx+kdx*i and j[i]+50>300 and x<kx+i*kdx+15:
        print "end"
        return "end"     
    else:
        return "go"
for i in range(nwall):
    wall.append(pygame.image.load("ml3stuff/wall.png").convert())
    wall[i]=pygame.transform.scale(wall[i], (15,50))
pygame.display.update()     
x=0
y=0
j=[100 for i in range(nwall)]
inc=[random.randint(4,12) for i in range(nwall)]
kx=80
kdx=50
uscita=0
niter=200
totdx=0
totdxold=0
primetime=0
limprimetime=15
dec=1
changedec=1
gen=1
font = pygame.font.SysFont("comicsansms", 72)

for t in range(niter):
    j=[100 for i in range(nwall)]    
    x=0
    uscita=0
    epressed=0
    primetime+=1
    print "primetime",primetime
 
    
    
    dec=changedec

    if primetime>limprimetime:
        w=[random.uniform(0,0.25) for i in range(4)]
        wold=[]
        wold[:]=w[:]
        gen+=1

        print "RESET"
        time.sleep(1)
        primetime=0
        todxold=0
        changedec=1
    if  totdx>totdxold:
        totdxold=totdx
        wold[:]=w[:]
        primetime=0    
    for s in range(len(w)):
        casual=random.uniform(-0.05*dec,0.05*dec)
        w[s]=wold[s]+casual
    if w[1]<0:
        w[1]=0
    text = font.render(str(gen), True, (0, 128, 0))
    
    while uscita==0:
 
        screen.blit(text,(10,10))        
        if dx==1:
            dx=100000
        for i in range(nwall):
           if x+wx<+kx+i*kdx:
               if abs(x+wx-(kx+i*kdx))<dx:
                   dx=abs(x+wx-(kx+i*kdx))
                   print dx
                   indx=i  
                   print "dy", dy

                   if inc[i]>0:
                       dw=1
                       up=0
                   else:
                       dw=0
                       up=1
        
        dy=300-j[indx]-50+wy
                       
        if neuron(dx,dy,up,dw)=="yes":               
            x+=2
        else:
            x+=0
            print "ARRESTED"
            #time.sleep(0.05)
            print dx,dy
        print     neuron(dx,dy,up,dw)
        print "dec",dec
        if x>200 and changedec==1:
            changedec=0.8
        if x>300 and changedec>0.5:
            changedec=0.5
        if x==600:
            changedec=.25
            x=0
            print "Well done",w
        keys = pygame.key.get_pressed()

        if keys[pygame.K_p]:
            time.sleep(1)

        pygame.display.update()
        pygame.event.pump()
        if keys[pygame.K_q]:
            pygame.quit()
        if keys[pygame.K_e]:
            uscita=1
            epressed=1
            print "e"
            keys[pygame.K_x]
            time.sleep(0.1)
             
 
        screen.blit(hero,(x,300))
        
            
        for i in range(nwall):
 
            if check_crash(i,x)=="end":
                uscita=1
                totdx=x
            
            j[i]+=inc[i]
            if j[i]<100:
                j[i]=100
                inc[i]=-inc[i]
            if j[i]>=300-50+wy:
                j[i]=300-50+wy
                inc[i]=-inc[i]

            screen.blit(wall[i],(+kx+i*kdx,j[i]))
        
     
        pygame.display.update()
        pygame.display.flip()
        screen.blit(background,(0,0))
        time.sleep(0.01)
    print w
print totdxold
print dx,dy
                
        #background cicle
        #shot cicle
        #monster cicle
