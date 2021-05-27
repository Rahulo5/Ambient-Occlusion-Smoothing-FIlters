import matplotlib.pyplot as plt

class my_dictionary(dict):
    
    def __init__(self):
        self = dict()
        
    def add(self, key, value):
        self[key] = value

f = open("data.txt", "r")
p = True
obj1 = my_dictionary()
u = ""
for x in f:
    if(p):
        t = x.split()
        u = t[1] + "_" + t[2] + "_" + t[3] + "_a"
        obj1.add(u,t[0])
    else:
        u = u + "b"
        obj1.add(u,x)
    p = not p

x = [5,10,15,20,25,30,35,40,45,50,55,60]
y = [1,2,3,4,5]
for i in y:
    for j in x:
        x_values1 = []
        y_values1 = []
        x_values2 = []
        y_values2 = []
        for k in x:
            y_values1.append(float(obj1[str(j)+ "_" + str(k) + "_" + str(i) + "_a"]))
            x_values1.append(k)
            y_values2.append(float(obj1[str(j)+ "_" + str(k) + "_" + str(i) + "_ab"]))
            x_values2.append(k)
        plt.plot(x_values1,y_values1)
        plt.plot(x_values2,y_values2)
        plt.xlabel("distance sigma")
        plt.ylabel("avg error from expected")
        plt.title("w = " + str(i) + " sigma normal constant = " + str(j))
        plt.savefig(str(i) + "_" + str(j) + ".png")
        plt.clf()