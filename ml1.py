#which number is odd which number is even
#n->wn->(odd,even)

import random
n=19
numbers=[(i+1) for i in range(n)]
weights=[0.5 for i in range(n)]
weightsnew=[]
weightsnew[:]=weights[:]
res=[0 for i in range(n)]
resold=[]
resold[:]=res[:]
exactres=[]
error=0
l1=0
l2=1
deltaw=[0 for i in range(n)]

def neuron(w,j):
    prob=random.uniform(0,1)
    if w>=0.5:
        res[j]=2
    else:
        res[j]=1

def check_result():
    error=0
    for k in range(len(numbers)):
        if res[k]%2!=numbers[k]%2:
            error+=1
    return error        
            
niter=5
errorold=len(numbers)
print (check_result())
print (res)

for i in range(niter):
    for j in range(len(weights)):
        neuron(weightsnew[j],j)

    if i==0:
        for k in range(9):
            casual=random.uniform(-0.2,0.2)
            weightsnew[k]+=casual
            if weightsnew[k]<l1:
                weightsnew[k]=l1
            if weightsnew[k]>l2:
                weightsnew[k]=l2

    if i>0:
        casual=random.uniform(0,1)
        if check_result()<=errorold:
            weights[:]=weightsnew[:]
            errorold=check_result()
            resold[:]=res[:]

        for k in range(n):
            casual=random.uniform(-0.2,0.2)
            weightsnew[k]=weights[k]+casual
            if weightsnew[k]<l1:
                weightsnew[k]=l1
            if weightsnew[k]>l2:
                weightsnew[k]=l2
                
                
         

     
print (weights)
print (resold)
print (errorold)
         
        
        
        
    
    
