import numpy as np
import os
from PIL import Image


def trained_network(nx,ny,typeofnn,numbneurons,noutputs,outputs):
    if typeofnn=='elusoft':
        for i in range(ny):
            for j in range(nx):
                index=i*nx+j
                inputs[index]=uni_matrix[i][j]

        matfirstweightstranspose=matfirstweights.transpose()
        for k in range(numbneurons):    
            sumfunction=np.sum(matfirstweightstranspose[k]*inputs)
            if sumfunction>=0:
                activfunc[k]=sumfunction
            else:
                activfunc[k]=np.exp(sumfunction)-1
        matsecondweightstranspose=matsecondweights.transpose()        
 
        for j in range(noutputs):
            outputs[j]=np.sum(matsecondweightstranspose[j]*activfunc)    
        print(outputs)    

    

def training_network(nx,ny,geomfile,typeofnn,numbneurons,noutputs,t,outputs):

    if typeofnn=='elusoft':
        for i in range(ny):
            for j in range(nx):
                index=i*nx+j
                inputs[index]=uni_matrix[i][j]

        matfirstweightstranspose=matfirstweights.transpose()
        for k in range(numbneurons):    
            sumfunction=np.sum(matfirstweightstranspose[k]*inputs)
            if sumfunction>=0:
                activfunc[k]=sumfunction
            else:
                activfunc[k]=np.exp(sumfunction)-1
        matsecondweightstranspose=matsecondweights.transpose()        
 
        for j in range(noutputs):
            outputs[j]=np.sum(matsecondweightstranspose[j]*activfunc)
###################### first differentiation matrix setting #########################
            
        for i in range(noutputs):
            for j in range(nx*ny):
                for k in range(numbneurons):
                    sumfunction=np.sum(matfirstweightstranspose[k]*inputs)
                    
                    if sumfunction>=0:
                        diffmatrixonfirstweights[i][j][k]=matsecondweights[k][i]*inputs[j]
                    if sumfunction<0:
                        diffmatrixonfirstweights[i][j][k]=matsecondweights[k][i]*np.exp(sumfunction)*inputs[j]

        for i in range(noutputs):
            for j in range(noutputs):
                for k in range(numbneurons):
                    sumfunction=np.sum(matfirstweightstranspose[k]*inputs)
                    if sumfunction>=0:
                        diffmatrixonsecondweights[i][k][j]=sumfunction
                    else:
                        diffmatrixonsecondweights[i][k][j]=(np.exp(sumfunction)-1)
                                
        sumonexpoj=np.sum(np.exp(outputs))


        if geomfile=='rec':
            s=0
        if geomfile=='tri':
            s=1
        if geomfile=='pen':
            s=2
        if geomfile=='exa':
            s=3
        for k in range(len(outputs)):
            if k!=s:
                totcost[0]+=(np.exp(outputs[k])/sumonexpoj)
            if k==s:
                totcost[0]+=(1-np.exp(outputs[k])/sumonexpoj)
                
        den=sumonexpoj**2
        
        ####### The gradient part
            
        for i in range(nx*ny):
            for j in range(numbneurons):
                for k in range(noutputs):
                    if k!=s:
                       gradcostonfirst[t][i][j]+=(np.exp(outputs[k])*diffmatrixonfirstweights[k][i][j]*sumonexpoj)
                       sumdiffter=0

                       for h in range(noutputs):
                           sumdiffter+=diffmatrixonfirstweights[h][i][j]*np.exp(outputs[h])
                       gradcostonfirst[t][i][j]-=np.exp(outputs[k])*sumdiffter


                    if k==s:
                       gradcostonfirst[t][i][j]-=(+np.exp(+outputs[k])*diffmatrixonfirstweights[k][i][j]*sumonexpoj)
                       sumdiffter=0
                       for h in range(noutputs):
                           sumdiffter+=diffmatrixonfirstweights[h][i][j]*np.exp(outputs[h])
                       gradcostonfirst[t][i][j]+=np.exp(outputs[k])*sumdiffter
        gradcostonfirst[t]/=den
                        
                        
        for i in range(numbneurons):
            for j in range(noutputs):
                for k in range(noutputs):
                    if k!=s:
                       gradcostonsecond[t][i][j]+=(np.exp(outputs[k])*diffmatrixonsecondweights[k][i][j]*sumonexpoj)
                       sumdiffter=0

                       for h in range(noutputs):
                           sumdiffter+=diffmatrixonsecondweights[h][i][j]*np.exp(outputs[h])
                       gradcostonsecond[t][i][j]-=np.exp(outputs[k])*sumdiffter


                    if k==s:
                       gradcostonsecond[t][i][j]-=(+np.exp(+outputs[k])*diffmatrixonsecondweights[k][i][j]*sumonexpoj)
                       sumdiffter=0
                       for h in range(noutputs):
                           sumdiffter+=diffmatrixonsecondweights[h][i][j]*np.exp(outputs[h])
                       gradcostonsecond[t][i][j]+=np.exp(outputs[k])*sumdiffter
        gradcostonsecond[t]/=den
                                               
                
                
    
##################################################################################        
        #print(sumfunction)
        #print(matfirstweights)
   # print (uni_matrix)
    
    

def matrix_scaling(nx,ny):
    minx=10**8
    maxx=0
    miny=10**8
    maxy=0
    for i in range(width):
        for j in range(height):
            mat[j][i]=int(im[i,j])
            if mat[j][i]==0:
                if i<=minx:
                    minx=i
                if i>=maxx:
                    maxx=i
                if j<=miny:
                    miny=j
                if j>=maxy:
                    maxy=j
    stepx=((maxx-minx)/nx)
    stepy=((maxy-miny)/ny)
    print (stepx,stepy)
    print(minx,miny)
    print (maxx,maxy)
    
    for i in range(ny):
        for j in range(nx):
            for l in range(int(round(i*stepy)),int(round(i+1)*stepy)):
                for k in range(int(round(j*stepx)),int(round(j+1)*stepx)):
                    if mat[miny+l][minx+k]==0:
                        uni_matrix[i][j]=1
    
    print(minx+(nx)*stepx,miny+(ny)*stepy)
   # print (uni_matrix)    
    mat[:][height-1]=0
##################################################

choice=input(" 1 - Training\n 2 - Trained\n ")
nx=32
ny=32
numbneurons=100
noutputs=4
inputs=np.zeros(nx*ny)
outputs=np.zeros(noutputs)
activfunc=np.zeros(numbneurons)
maxsetmatfirst=2*np.sqrt(2)*np.sqrt(6/(nx*ny+numbneurons))
maxsetmatsec=2*np.sqrt(2)*np.sqrt(6/(noutputs+numbneurons))
typeofnn='elusoft'
niterations=110
learningpar=0.01
tollerance=2.0
totcost=[0]

print (inputs)

if choice=='1':
    f1=open('mat1.dat','w')
    f2=open('mat2.dat','w')
    maxsetmax=0 ##########à da settare #############
    matfirstweights=np.random.rand(nx*ny,numbneurons)*maxsetmatfirst-maxsetmatfirst/2
    matsecondweights=np.random.rand(numbneurons,noutputs)*maxsetmatsec-maxsetmatsec/2
    diffmatrixonfirstweights=np.zeros((noutputs,nx*ny,numbneurons))
    diffmatrixonsecondweights=np.zeros((noutputs,numbneurons,noutputs))

    fileslist=[]
    fileslist=os.listdir()
    pnglistfiles=[]
    for i in range(niterations):
        totfirstgrad=np.zeros((nx*ny,numbneurons))
        totsecgrad=np.zeros((numbneurons,noutputs))        
        totcost[0]=0
        t=0
        if i==0:
            for el in fileslist:
                try:
                    if el.split('.')[1]=='png':
                        pnglistfiles.append(el)
                    
                except:
                    pass
        gradcostonfirst=np.zeros((len(pnglistfiles),nx*ny,numbneurons))
        gradcostonsecond=np.zeros((len(pnglistfiles),numbneurons,noutputs))     
        for el in pnglistfiles:
            print (el[0:3])
            if (el[0:3]=='rec') or (el[0:3]=='tri') or (el[0:3]=='pen') or (el[0:3]=='exa'):
                shape = Image.open(el)
                img = shape.convert('L')
                im = img.load()
                width, height = img.size
                mat=np.zeros((height,width))
                uni_matrix=np.zeros((ny,nx))
                matrix_scaling(nx,ny)
                training_network(nx,ny,el[0:3],typeofnn,numbneurons,noutputs,t,outputs)
                t+=1
                print (np.exp(outputs)/np.sum(np.exp(outputs)))
        for j in range(len(pnglistfiles)):
            totfirstgrad+=gradcostonfirst[j]
            totsecgrad+=gradcostonsecond[j]
            
        matfirstweights-=learningpar*totfirstgrad
        matsecondweights-=learningpar*totsecgrad
        print ("the total cost is",totcost[0])
        print (totfirstgrad)
        if totcost[0]<=tollerance:
            break
        
    for i in range(nx*ny):
        for j in range(numbneurons):
            f1.write(str(matfirstweights[i][j])+str(','))
        f1.write(str('\n'))
    for i in range(numbneurons):
        for j in range(noutputs):
            f2.write(str(matsecondweights[i][j])+str(','))
        f2.write(str('\n'))
    f1.close()
    f2.close()

if choice=='2':
    filename=input('Insert the name of the image file=')
    matfirstweights=np.random.rand(nx*ny,numbneurons)*maxsetmatfirst-maxsetmatfirst/2
    matsecondweights=np.random.rand(numbneurons,noutputs)*maxsetmatsec-maxsetmatsec/2
    f1=open('mat1.dat','r')
    f2=open('mat2.dat','r')
    nline=0
    string=''
    listval=[]
    for line in f1:
        #print(nline)
        #print (line)
        listval=line.split(',')
        listval[-1:]=[]
        #print (len(listval))
        for i in range(numbneurons):
            matfirstweights[nline][i]=listval[i]
        nline+=1
    nline=0    
    for line in f2:
        #print(nline)
        #print (line)
        listval=line.split(',')
        listval[-1:]=[]
        #print (len(listval))
        for i in range(noutputs):
            matsecondweights[nline][i]=listval[i]
            
        nline+=1
    shape = Image.open(filename)
    img = shape.convert('L')
    im = img.load()
    width, height = img.size
    mat=np.zeros((height,width))
    uni_matrix=np.zeros((ny,nx))
    matrix_scaling(nx,ny)    
    trained_network(nx,ny,typeofnn,numbneurons,noutputs,outputs)        
    print (np.exp(outputs)/np.sum(np.exp(outputs)))        

#print (uni_matrix)        
        
