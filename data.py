import sys, json
import math  
import random 
import numpy as np
import os.path

######################################################
##### Skript zum generieren der Labels
##### Erstellt JSON aller Lagen 
######################################################

### Initialisierung 

t = 300 # Teiler: t=10 -> 1002, t=400 -> 1.6 Millionen
objRad = 90 # Abstand
fov = 60
res = [1196,1196]
filename = os.path.join("data", "amarok_tuer.json")

usageSplit = [0, 0.7, 0.9] #70% train, 20% valid, 10% test

#Bereichsbeschraenkungen
rangeR = [objRad,3*objRad]
rangeRoll = [-45,45]

### Programm

phi = 0.5*(1+math.sqrt(5))

#Ikosaeder Koordinaten
c = [
	[0,-1,phi],
	[0,1,phi],
	[0,-1,-phi],
	[0,1,-phi],
	[-1,phi,0],
	[1,phi,0],
	[-1,-phi,0],
	[1,-phi,0],
	[phi,0,-1],
	[phi,0,1],
	[-phi,0,-1],
	[-phi,0,1]
	]

#Ikosaeder Flaechen	
planes = [ 
	[c[4],c[10],c[11]],
	[c[10],c[11],c[6]],
	[c[6],c[11],c[0]],
	[c[11],c[0],c[1]],
	[c[0],c[1],c[9]],

	[c[9],c[1],c[5]],
	[c[1],c[5],c[4]],
	[c[5],c[4],c[3]],
	[c[3],c[4],c[10]],
	[c[8],c[9],c[5]],

	[c[2],c[10],c[3]],
	[c[7],c[0],c[9]],

	#Ab hier ohne Startpunkt
	[c[6],c[7],c[2]],
	[c[7],c[8],c[2]],
	[c[8],c[2],c[3]],

	#Ab hier ohne Kanten
	[c[5],c[3],c[8]],
	[c[4],c[1],c[11]],
	[c[2],c[10],c[6]],
	[c[6],c[7],c[0]],
	[c[9],c[7],c[8]]
]

v = []

dl = 2/t

def vectAdd(v1,b,v2):
	res = [v1[0]+b*v2[0],v1[1]+b*v2[1],v1[2]+b*v2[2]]
	return res

def length(v):
	return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

#Formatiert Winkel zwischen -180,180
def fa(angle): 
    angle %= 360
    if(angle>180):
        angle -= 360;
    return angle

#Waehlt nach Werten in split Eintrag aus String[]
def chooseRandomly(str,split):
	rand = random.random()
	for i in range(len(str)):
		if rand > split[i]:
			result = str[i]
	return result

#Berechnet pitch,yaw	
def calcAngles(v):
    pitch = math.degrees(math.atan2(v[1],math.sqrt(v[0]*v[0]+v[2]*v[2])))
    yaw = 180+math.degrees(math.atan2(v[0],v[2]))
    return pitch,yaw

#Fuegt Koordiantenpunkt hinzu
def add(vector,plane,s={'n':0},first={'b':True}):
	randR = random.uniform(rangeR[0],rangeR[1])
	
	randAngle = random.uniform(0,2*math.pi)
	randRadius = random.random()*fov/2 
	randPitch = randRadius*math.sin(randAngle)
	randYaw = randRadius*math.cos(randAngle)
	randRoll = random.uniform(rangeRoll[0],rangeRoll[1])
	
	
	x = randR*vector[0]/length(vector)
	y = randR*vector[1]/length(vector)
	z = randR*vector[2]/length(vector)
		
	pose = np.array([x,y,z,0,0,0]) 

	angleToCenter = calcAngles(pose[:3])
	
	pose[3] = angleToCenter[0]+randPitch
	pose[4] = angleToCenter[1]+randYaw
	pose[5] = randRoll
	
	seq = s['n']

	if first['b']==False:
		file.write(",")
	else:
		first['b'] = False
	
	usage = chooseRandomly(["train","valid","test"],usageSplit)

	file.write("\n\t{\n\t\t")
	file.write("\"name\": \"img_"+str("%08i" % seq)+".png\",\n")
	file.write("\t\t\"x\": "		+str(pose[0])+	",\n")
	file.write("\t\t\"y\": "		+str(pose[1])+	",\n")
	file.write("\t\t\"z\": "		+str(pose[2])+	",\n")
	file.write("\t\t\"pitch\": "	+str(pose[3])+",\n")
	file.write("\t\t\"yaw\": "	    +str(pose[4])+	",\n")
	file.write("\t\t\"relpitch\": "	+str(randPitch)+",\n")
	file.write("\t\t\"relyaw\": "	+str(randYaw)+	",\n")
	file.write("\t\t\"roll\":"		+str(pose[5])+	",\n")
	file.write("\t\t\"usage\": \""	+usage+		"\"\n")
	file.write("\t}")
		
	s['n']+=1

with open(filename,'w') as file: 
	file.write("{\n\t\"resolution\": ["+str(res[0])+","+str(res[1])+"],\n")
	file.write("\t\"norm\": ["+str(rangeR[1])+","+str(rangeR[1])+","+str(rangeR[1])+","+str(fov/2)+","+str(fov/2)+","+str(rangeRoll[1])+"],\n")
	file.write("\t\"position\": [")

	for j in range(len(planes)):
		print("Create plane "+str(j)+"\n")
		a = planes[j][0]
		b = planes[j][1]
		c = planes[j][2]

		if j < 12:
			add(a,j)

		for i in range(1,t):
			ab = vectAdd(b,-1,a) #vector from point A to point B
			ac = vectAdd(c,-1,a) #vector from point A to point C	

			d = vectAdd(a,i/t,ab) # next point on line from A to B
			e = vectAdd(a,i/t,ac) # next point on line from A to C

			if j < 15:
				add(d,j) 
				add(e,j)
				
			if i>1:
				de = vectAdd(e,-1,d) #vector from point D to point E
				for m in range(1,i):
					f = vectAdd(d,m/i,de) #next point on line from D to E
					add(f,j)

	file.write("\n\t]\n}")

#Test	
with open(filename) as f:
	num = 0
	splits = [0,0,0]
	js = json.load(f)
	for pos in js['position']:
		num = num+1
		
		if pos['usage'] == "train":
			splits[0] = splits[0] + 1
		elif pos['usage'] == "valid":
			splits[1] = splits[1] + 1
		else:
			splits[2] = splits[2] + 1
		

	print("positions:",num)
	print("max size:",num*res[0]*res[1]*4/1024/1024/1024,"GB")
	print("train:",splits[0])
	print("valid:",splits[1])
	print("test:",splits[2])			