import random

    

#*****winning configurations*********
listwinning=['001010001','000010101','101010000','100010100']
listwinning=['000010001','010100000','100010000','000010100','001010000','000001010','000100010','000100010','010001000']
listwinstranslate=[]
statpos=[0 for i in range(2**9)]
listnumbers=[i+1 for i in range(9)]
print (listwinning)

for value in listwinning:
    
    listwinstranslate.append(((int(value,2))))
print (listwinstranslate)
ntrial=200

#****start the process***************
def print_solutions():
    for value in numtotable:
        print("\n")
        cont=-1
        for i in range(3):
            #print("\n")
            print(value[i*3:(i+1)*3])
            


def test_it():
    agree=1
    binarystring=""
    youpass=0
    for i in range(3):
        for j in range(3):
            binarystring+=str(playermatrix[i][j])
    binarystring=binarystring[::-1]
   
    for i in range(len(listwinning)):
        agree=1
    
        if youpass==0:
            for j in range(len(binarystring)):
                
                if (int(binarystring[j])==0 and int(listwinning[i][j])==1):
                    agree=0
            if (agree==1):
                youpass=1
    
                
    return [youpass,binarystring]        
        
    
            
            
    
def filling_in():
    vv=random.randint(0,len(listnumbers)-1)
    v=listnumbers[vv]
    del listnumbers[vv]
    x=-1
    y=0
    for j in range(v):
        x+=1
        
        if x>2:
            x=0
            y+=1
       
         
    playermatrix[y][x]=1
    print ((x,y,v))
win='false'
cont=0
for i in range(ntrial):

    listnumbers=[k+1 for k in range(9)]    
    playermatrix=[[0 for k in range(3)] for j in range(3)]
    win='false'
    while win=='false':
        filling_in()
        if test_it()[0]==1:
            win='true'
            print (win)
            print ("and")
            print (test_it()[1])
            statpos[int(test_it()[1],2)]+=1
            nocount=1
            for i in range(3):
                for j in range(3):
                    if playermatrix[i][j]!=1:
                        nocount=0
            if nocount==1:
                
                cont+=1
                
        else:
            statpos[int(test_it()[1],2)]-=1

        #win='true'    

print (playermatrix)        
print (statpos)
print (listwinstranslate)
print (statpos[81],statpos[21],statpos[336],statpos[276])
print (cont)
i=-1
memvalue=[]
for value in statpos:
    i+=1
    if value>0:
        memvalue.append(i)
tot=0
maxim=10
for value in memvalue:
    num=bin(value)
    tot=0
    print (num)
    for n in num:
        if n=='1':
            tot+=1
            
    if tot<maxim:
        maxim=tot

memtab=[]

for value in memvalue:
    num=bin(value)
    tot=0
    print (num)
    for n in num:
        if n=='1':
            tot+=1
            
    if tot==maxim:
        print (value)
        memtab.append(value)
print ("********")
numtotable=[]

for value in memtab:
    num=bin(value)
    num=num[2:]
  
    zerostring=""
    if len(num)<9:
        for k in range(len(num),9):
            zerostring+=str("0")
    zerostring+=str(num)
    print (zerostring)
    numtotable.append(zerostring)

print_solutions()    
        
print (maxim)

