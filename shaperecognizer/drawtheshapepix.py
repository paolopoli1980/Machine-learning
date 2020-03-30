import numpy as np
import pygame
import sys
import time
from tkinter import *
from pygame.locals import *


pygame.init()
size = width, height = 100, 100
speed=[]
screen = pygame.display.set_mode(size)
background = pygame.image.load('Immagine.png').convert()
screen.blit(background, (0, 0))

   

def call_win(dl,limplayx,limplayy,width,height):
    
    def clicked_triangle():

       # lbl.configure(text="Button was clicked !!")
        
        pygame.image.save(screen,"triangle"+str(listofshapes[0])+str(".png"))
       
        listofshapes[0]+=1
        
        mempoint[:]=[]
        memlines[:]=[]
        
    def clicked_rectangle():
        pygame.image.save(screen,"rectangle"+str(listofshapes[1])+str(".png"))
        
        listofshapes[1]+=1
        
       # lbl.configure(text="Stop !!")
        
        
        mempoint[:]=[]
        memlines[:]=[]
        
    def clicked_trap():
        pygame.image.save(screen,"trap"+str(listofshapes[2])+str(".png"))
       
        listofshapes[2]+=1
        
       # lbl.configure(text="Stop !!")
        mempoint[:]=[]
        memlines[:]=[]
        
    def clicked_penta():
        pygame.image.save(screen,"penta"+str(listofshapes[3])+str(".png"))
       
        listofshapes[3]+=1
        
       # lbl.configure(text="Stop !!")
        mempoint[:]=[]
        memlines[:]=[]
    def clicked_exa():

        pygame.image.save(screen,"exa"+str(listofshapes[4])+str(".png"))
       
        listofshapes[4]+=1
        
       # lbl.configure(text="Stop !!")
        mempoint[:]=[]
        memlines[:]=[]        

    def clicked_refresh():

       # lbl.configure(text="Stop !!")
        mempoint[:]=[]
        memlines[:]=[]

    def clicked_finish():
        for i in range(len(listofshapes)):
            for j in range(listofshapes[i]):
                f1.write(str(i)+"\n")
                
        f1.close()        
        sys.exit()   
                
            
            
    window = Tk()
     
    window.title("Welcome to LikeGeeks app")
    pygame.display.flip()
    win=0 
    window.geometry('350x200')
   # lbl = Label(window, text="Hello")
   # lbl.grid(column=0, row=0)

    btn = Button(window, text="Triangle",command=clicked_triangle)
    btn.grid(column=0, row=0)
    btn = Button(window, text="Rectangle",command=clicked_rectangle)
    btn.grid(column=1, row=0)
    #btn = Button(window, text="Trapeze",command=clicked_trap)
    #btn.grid(column=2, row=0)
    btn = Button(window, text="Penta",command=clicked_penta)
    btn.grid(column=2, row=0)
    btn = Button(window, text="Exa",command=clicked_exa)
    btn.grid(column=3, row=0)  
    btn = Button(window, text="Refresh",command=clicked_refresh)
    btn.grid(column=4, row=0)
    btn = Button(window, text="Finish",command=clicked_finish)
    btn.grid(column=5, row=0)
    window.mainloop()


limplayx=20
limplayy=20
dl=20
white=(255,255,255)
npressed=0
mempoint=[]
memlines=[]
   
listofshapes=[0,0,0,0,0]
f1=open("list.AI","w")

while True:

  
    pygame.display.flip()   

    
    #time.sleep(0.1)
    for event in pygame.event.get():
        if event.type!= KEYDOWN:
            screen.fill(white)
        if event.type == pygame.QUIT: sys.exit()

        if event.type == KEYDOWN:
 
            tasti_premuti = pygame.key.get_pressed()        

            
            if event.key == K_w:
                call_win(dl,limplayx,limplayy,width,height)

        
        if len(mempoint)==1:
            
           
            pygame.draw.line(screen,(0,0,0),(mempoint[0][0],mempoint[0][1]),(pygame.mouse.get_pos()[0],pygame.mouse.get_pos()[1]),2)

        for el in memlines:
            pygame.draw.line(screen,(0,0,0),(el[0]),(el[1]),2)                        

        if (event.type == pygame.MOUSEBUTTONDOWN):
                            
            x=pygame.mouse.get_pos()[0]
            y=pygame.mouse.get_pos()[1]
            mempoint.append([x,y])
           # x=(width/2-limplayx*dl/2.0+dl*int((pygame.mouse.get_pos()[0]-(width/2-limplayx*dl/2.0))/dl))
           # y=(height/2-limplayy*dl/2.0+dl*int((pygame.mouse.get_pos()[1]-(height/2-limplayy*dl/2.0))/dl))
            if len(mempoint)==2:
                pygame.draw.line(screen,(0,0,0),(mempoint[0][0],mempoint[0][1]),(mempoint[1][0],mempoint[1][1]),2)
                memlines.append([[mempoint[0][0],mempoint[0][1]],[mempoint[1][0],mempoint[1][1]]])
                mempoint=[]
            
            print (x,y)
            print (memlines)

      

        
