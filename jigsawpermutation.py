import numpy as np
from sympy.combinatorics import permutations
from scipy import linalg

corners =np.array([
    [40,0],
    [7,72],
    [79,39],
    [32,47]])

edges = np.array([
    [1,48,2],
    [3,56,4],
    [5,64,6],
    [73,15,74],
    [75,23,76],
    [77,31,78],
    [38,71,37],
    [36,63,35],
    [34,55,33],
    [46,24,45],
    [44,16,43],
    [42,8,41]])

faces=np.array([
    [9,50,10,49],
    [11,58,12,57],
    [13,66,14,65],
    [17,52,18,51],
    [19,60,20,59],
    [21,68,22,67],
    [25,54,26,53],
    [27,62,28,61],
    [29,70,30,69]])

Z=np.concatenate([corners.flatten(),edges.flatten(),faces.flatten()])
Y=np.argsort(np.concatenate([corners.flatten(),edges.flatten(),faces.flatten()]))
Q=np.arange(80) +np.repeat([[1,-1]],40,axis=0).flatten()
class shuffling:
    def __init__(self):
        self.cornerperm = np.random.permutation(4)
        self.edgeperm =np.random.permutation(12)
        self.faceperm=np.random.permutation(9)

        self.facerotate=np.random.randint(4, size=9)


    def build_jigsaw(self):
        ncorners=corners[self.cornerperm]
        nedges=edges[self.edgeperm]
        nfaces=faces[self.faceperm]
        for i in range(9):
            nfaces[i]=np.roll(nfaces[i],self.facerotate[i])
        self.cutperm=np.concatenate([ncorners.flatten(),nedges.flatten(),nfaces.flatten()])[Y]
        self.alpha =Q[np.argsort(self.cutperm)[Q[self.cutperm]]]
        self.jigsawcuts=np.zeros(80,dtype=int)
        badness=0
        for (i,elt) in enumerate(permutations.Permutation(self.alpha).full_cyclic_form):
            if i%2==0:
                self.jigsawcuts[elt]=i//2+1
                if len(elt)==1:
                    badness+=1
            else:
                self.jigsawcuts[elt]=-(i//2)-1
        return Jigsaw(
            self.jigsawcuts[corners],self.jigsawcuts[edges],self.jigsawcuts[faces],badness)


class Jigsaw:
    def __init__(self,jcorners,jedges,jfaces,badness):
        self.jcorners=jcorners
        self.jedges=jedges
        self.jfaces=jfaces
        self.badness=badness
    def canonise_and_check_uniqueness(self):
        self.corners=sorted(set([tuple(corner) for corner in self.jcorners]))
        if len(self.corners)<4:
            return False
        self.edges=sorted(set([tuple(edge) for edge in self.jedges]))
        if len(self.edges)<12:
            return False
        
        self.faces =  sorted(set([tuple(np.roll(self.jfaces[i//4],i%4)) for i in range(36)]))
        if len(self.faces)<36:
            return False
        return True
                     
        
def gen_puzzle(N=1e5):
    att=0
    t=0
    jigsaws=[]
    for i in range(N):
        Q=shuffling()
        J=Q.build_jigsaw()
        if J.badness!=0:
            continue
        if J.canonise_and_check_uniqueness():
            att+=1
            g=gen_solutions(J)
            print(g)
            if g==2:
                return J

        
def gen_solutions(Jigsaw):
    #first generate the rim
    sol=np.array([[0]])
    crem=np.ones((1,4))
    crem[0,0]=0
    erem=np.ones((1,12))
    le=-np.array([Jigsaw.jcorners[0,-1]])
    def add_edge(sol,crem,erem,le):   
        u,v=np.nonzero((le.reshape(-1,1)==Jigsaw.jedges[:,0])*erem)    
        newsol=np.hstack([sol[u],v.reshape(-1,1)])
        newcrem=crem[u]
        newerem=erem[u]-(np.arange(12)==v.reshape(-1,1))
        newle=-Jigsaw.jedges[v][:,-1]
        return newsol,newcrem,newerem,newle
    def add_corner(sol,crem,erem,le):    
        u,v=np.nonzero((le.reshape(-1,1)==Jigsaw.jcorners[:,0])*crem)
        newsol=np.hstack([sol[u],v.reshape(-1,1)])
        newerem=erem[u]
        newcrem=crem[u]-(np.arange(4)==v.reshape(-1,1))
        newle=-Jigsaw.jcorners[v][:,-1]
        return newsol,newcrem,newerem,newle
    A=(sol,crem,erem,le)
    for i in range(3):
        for i in range(3):
             A=add_edge(*A)
        A=add_corner(*A)
    for i in range(3):
         A=add_edge(*A)
    loop=-Jigsaw.jedges[A[0][:,[1,2,3,5,6,7,9,10,11,13,14,15]],1]
    s1=np.zeros((loop.shape[0],0))
    r1=np.zeros((loop.shape[0],0))
    frem=np.ones((loop.shape[0],9))
    longjfaces= np.vstack([linalg.circulant(Jigsaw.jfaces[i]) for i in range(9)])
    def add_centre(loop,s1,r1,frem,start,end):
        l=end-start
        u,v=np.nonzero(np.repeat(frem,4,axis=1)*np.prod(np.expand_dims(loop[:,start:end],axis=1)==longjfaces[:,:l],axis=2))
        newloop=np.hstack([loop[u,:start],-np.flip(longjfaces[v,l:],axis=1),loop[u,end:]])
        news1=np.hstack([s1[u],(v//4).reshape(-1,1)])
        newr1=np.hstack([r1[u],(v%4).reshape(-1,1)])
        newfrem=frem[u]-(np.arange(9)==(v//4).reshape(-1,1))
        return newloop,news1,newr1,newfrem
    A=(loop,s1,r1,frem)
    A=add_centre(*A,5,7)
    A=add_centre(*A,6,8)
    A=add_centre(*A,7,10)
    A=add_centre(*A,4,6)
    A=add_centre(*A,5,7)
    A=add_centre(*A,6,9)
    A=add_centre(*A,2,5)
    A=add_centre(*A,1,4)
    A=add_centre(*A,0,4)
    return A[0].shape[0]
        
        
        


       
