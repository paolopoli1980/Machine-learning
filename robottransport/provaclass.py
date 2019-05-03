
class x:
    def self__init(self):
        self.t=0

    def met(self):
        for i in range(2):
            print (y[i].t)
        print (w)


w=[1,2,3]
y=[x(),x()]
#y.append(x())
y[0].t=1
y[1].t=99

y[0].met()
