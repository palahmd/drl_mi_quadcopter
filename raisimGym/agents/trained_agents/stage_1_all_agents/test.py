import os

directory = os.path.dirname(os.path.realpath(__file__))
data = []
for file in os.listdir(directory):
    if file.endswith(".pt"):
    	data.append(int(file.split("_", 1)[1].split(".", 1)[0]))
	

data.sort()
print(data[-1])
