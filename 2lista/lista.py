#!/usr/bin/env python3
# Author: Marco Thome

import os
import numpy as np
from PIL import Image
import math

# Helper
# i = Image.fromarray([[[R,G,B], [R,G,B], [R,G,B], [R,G,B]]])

# converts RGB numpy array to HSI numpy array
def rgbtohsv(im):
	if(np.any(im > 1)):
		im = im / 255.0
	R = im[:,:,0]
	G = im[:,:,1]
	B = im[:,:,2]
	# theta = math.acos(0.5*((R-G)+(R-B))/np.sqrt(0.0001+np.square(R-G)+(R-B)*(G-B)))
	# H = theta
	H = np.arccos(0.5*((R-G)+(R-B))/(0.00000001+np.sqrt(np.square(R-G)+((R-B)*(G-B)))))*180/math.pi
	H[B > G] = 360 - H[B > G]
	S = 1 - 3*np.minimum(R,np.minimum(G,B))/(R+G+B)
	V = (R+G+B)/3
	return np.stack([H,S,V],axis=2)
	# return np.stack([H/360.0,S,I],axis=2)

# converts HSI numpy array to RGB numpy array
def hsvtorgb(im):
	H = im[:,:,0]
	if(np.all(H <= 1)):
		H *= 360
	S = im[:,:,1]
	I = im[:,:,2]
	R = np.zeros(H.shape)
	G = np.zeros(H.shape)
	B = np.zeros(H.shape)
	tabela = np.logical_and(H>=0,H<120)
	B[tabela] = I[tabela]*(1-S[tabela])
	R[tabela] = I[tabela]*(1+(S[tabela]*np.cos(H[tabela]*math.pi/180))/(np.cos((60-H[tabela])*math.pi/180)))
	G[tabela] = 3*I[tabela]-(R[tabela]+B[tabela])
	tabela = np.logical_and(H>=120,H<240)
	H[tabela] -= 120
	R[tabela] = I[tabela]*(1-S[tabela])
	G[tabela] = I[tabela]*(1+(S[tabela]*np.cos(H[tabela]*math.pi/180))/(np.cos((60-H[tabela])*math.pi/180)))
	B[tabela] = 3*I[tabela]-(R[tabela]+G[tabela])
	tabela = np.logical_and(H>=240,H<360)
	H[tabela] -= 240
	G[tabela] = I[tabela]*(1-S[tabela])
	B[tabela] = I[tabela]*(1+(S[tabela]*np.cos(H[tabela]*math.pi/180))/(np.cos((60-H[tabela])*math.pi/180)))
	R[tabela] = 3*I[tabela]-(B[tabela]+G[tabela])
	# R = np.rint(R*255)
	# G = np.rint(G*255)
	# B = np.rint(B*255)
	return np.rint(np.stack([R,G,B],axis=2)*255).astype(np.uint8())

def erodir(im,es=None,size=5,dot=False):
	res = np.zeros(im.shape, dtype=np.uint8)
	if(es is None):
		side = size//2
		for i in range(im.shape[0]):
			for j in range(im.shape[1]):
				sti = i - side if i > side else 0
				eni = i + side if i <= (im.shape[0]-side) else im.shape[0]
				stj = j - side if j > side else 0
				enj = j + side if j <= (im.shape[1]-side) else im.shape[1]
				if(dot and (np.sum(im[sti:eni,stj:enj]) == im[i,j])):
					res[i,j] = im[i,j]
				else:
					res[i,j] = np.min(im[sti:eni,stj:enj])
	return res

def dilatar(im,es=None,size=5):
	res = np.zeros(im.shape, dtype=np.uint8)
	if(es is None):
		side = size//2
		for i in range(im.shape[0]):
			for j in range(im.shape[1]):
				sti = i - side if i >= side else 0
				eni = i + side if i < (im.shape[0]-side) else im.shape[0]
				stj = j - side if j >= side else 0
				enj = j + side if j < (im.shape[1]-side) else im.shape[1]
				res[i,j] = np.max(im[sti:eni,stj:enj])
	return res

# returns P1(k) = sum i=0..k pi | pi=ni/MN
# l = levels in image
# n = number of pixels according to each level
def P12(l,n,MN,k):
	p1 = np.sum(n[l <= k])/MN
	return (p1,1-p1)

# returns (1/P1(k))* sum i=0..k i*pi
# def m12(l,n,MN,k,M2=False):
# 	P12 = P12(l,n,MN,k)
# 	tab = l <= k
# 	m1 = np.sum(l[tab]*n[tab])/(MN*P12[0])
# 	if(M2):
# 		tab = l > k
# 		m2 = np.sum(l[tab]*n[tab])/(MN*P12[1])
# 		return (P12[0],m1,P12[1],m2)
# 	else:
# 		return (P12[0],m1)

def mkg(l,k,MN,ip):
	return (np.sum(ip[l<=k])/MN, np.sum(ip)/MN)

def otsu(im):
	levels,number = np.unique(im,return_counts=True)
	MN = im.size
	ip = levels*number
	varb = np.zeros(levels.shape[0]-1)
	for i in range(levels.shape[0]-1):
		k = levels[i]
		P1,_ = P12(levels,number,MN,k)
		mk,mg = mkg(levels,k,MN,ip)
		varb[i] = np.square(mg*P1-mk)/(P1-P1*P1)
	return levels[np.argmax(varb)]

def laplaciano(im):	# binary image
	res = np.zeros(im.shape, dtype=np.bool)
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			fx1 = im[i - 1,j].astype(np.uint16) if i > 0 else 0
			fx2 = im[i + 1,j].astype(np.uint16) if i < (im.shape[0]-1) else 0
			fy1 = im[i,j - 1].astype(np.uint16) if j > 0 else 0
			fy2 = im[i,j + 1].astype(np.uint16) if j < (im.shape[1]-1) else 0
			res[i,j] = (fx1 + fx2 + fy1 + fy2) > (4*im[i,j])
	return (res*255).astype(np.uint8)

def hough(im):
	D = int(np.ceil(np.sqrt(im.shape[0]*im.shape[0]+im.shape[1]*im.shape[1])))
	pth = np.zeros((501,4*181))
	args = np.argwhere(im > 250)
	for x,y in args:
		for t in range(4*181):
			th = (t/4)-91
			p = int((np.rint(x*np.cos(th*math.pi/180) + y*np.sin(th*math.pi/180))+D)*250/D)
			pth[p,t] += 1
	return ((pth / pth.max())*255).astype(np.uint8)

def limiarizar(im,val,inv=False):
	if(inv):
		return ((im < val)*255).astype(np.uint8)
	else:
		return ((im > val)*255).astype(np.uint8)

def crescer(im,cond):
	loop = True
	while(loop):
		loop = False
		for i in range(im.shape[0]):
			for j in range(im.shape[1]):
				if(im[i,j] > 200):
					continue
				else:
					sti = i - 10 if i >= 10 else 0
					eni = i + 10 if i < (im.shape[0]-10) else im.shape[0]
					stj = j - 10 if j >= 10 else 0
					enj = j + 10 if j < (im.shape[1]-10) else im.shape[1]
					if((np.max(im[sti:eni,stj:enj])>200) and (cond[i,j] < 68)):
						im[i,j] = 255
						loop = True
	return im

def m2d(im,p,q):	# momento 2D de ordem (p+q)
	n = im.astype(np.int)
	if((p+q)==0):
		return n.sum()
	for i in range(n.shape[0]):
		for j in range(n.shape[1]):
			n[i,j] *= math.pow(i,p)*math.pow(j,q)
	return n.sum()

def mc2d(im,p,q):	# momento central 2D de ordem (p+q)
	n = im.astype(np.int)
	if((p+q)==0):
		return n.sum()
	xm = m2d(im,1,0)/n.sum()
	ym = m2d(im,0,1)/n.sum()
	for i in range(n.shape[0]):
		for j in range(n.shape[1]):
			n[i,j] *= math.pow(i-xm,p)*math.pow(j-ym,q)
	return n.sum()

def mcn(im,p,q):	# momentos centrais normalizados de ordem (p+q)
	return mc2d(im,p,q)/math.pow(mc2d(im,0,0),1+(p+q)/2)

def moments(im):
	m20 = mcn(im,2,0)
	m02 = mcn(im,0,2)
	m11 = mcn(im,1,1)
	m12 = mcn(im,1,2)
	m21 = mcn(im,2,1)
	m30 = mcn(im,3,0)
	m03 = mcn(im,0,3)
	print('1st:', m20 + m02)
	print('2nd:', math.pow(m20 - m02,2) + 4*math.pow(m11,2))
	print('3rd:', math.pow(m30 - 3*m12,2) + math.pow(3*m21 - m03,2))
	print('4th:', math.pow(m30 + m12,2) + math.pow(m21 - m03,2))
	print('5th:', (m30 - 3*m12) * (m30 + m12) * (math.pow(m30 + m12,2) - 3*math.pow(m21 + m03,2)) + (3*m21 - m03) * (m21 + m03) * (3*math.pow(m30 + m12,2) - math.pow(m21 + m03,2)))
	print('6th:', (m20 - m02) * (math.pow(m30 + m12,2) - math.pow(m21 + m03,2)) + 4*m11 * (m30 + m12) * (m21 + m03))
	print('7th:', (3*m21 - m03)*(m30 + m12)*(math.pow(m30 + m12,2) - 3*math.pow(m21 + m03,2)) + (3*m12 - m30)*(m21 + m03)*(3*math.pow(m30 + m12,2) - math.pow(m21 + m03,2)))

def pbranco(im):
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			if(im[i,j] > 225):
				return (i,j)

# 3  2  1
# 4  X  0
# 5  6  7
def fronteira(im):
	i,j = pbranco(im)	# search for the first white dot
	b = [(i,j)]
	prev = 3
	while(True):
		prev = (prev + 1) % 8
		if((i == 0) and (prev >= 3) and (prev <= 5)):
			prev = 6
		elif((i == (im.shape[0] - 1)) and ((prev == 7) or (prev == 0) or (prev == 1))):
			prev = 2
		if((j == 0) and (prev >= 1) and (prev <= 3)):
			prev = 4
			if(i == 0):
				prev = 6
		elif((j == (im.shape[1] - 1)) and (prev >= 5) and (prev <= 7)):
			prev = 0
			if(i == (im.shape[0] - 1)):
				prev = 2
		tup = (i,j)
		if(prev == 0):
			tup = (i+1,j)
		elif(prev == 1):
			tup = (i+1,j-1)
		elif(prev == 2):
			tup = (i,j-1)
		elif(prev == 3):
			tup = (i-1,j-1)
		elif(prev == 4):
			tup = (i-1,j)
		elif(prev == 5):
			tup = (i-1,j+1)
		elif(prev == 6):
			tup = (i,j+1)
		elif(prev == 7):
			tup = (i+1,j+1)
		else:
			print('error:',b,prev)
			return b
		if(im[tup] >= 225):
			if(b[0] == tup):
				return b
			elif(tup in b):
				print("error:",b,tup)
				return b
			else:
				b.append(tup)
				i,j = tup
				prev = (prev + 4) % 8
	return b

def complexar(v):
	return np.array([complex(i,j) for i,j in v])

def descomplexar(v):
	return np.array([(int(i.real),int(i.imag)) for i in v])

def descritores(v,inv=False,P=0):
	if(inv):
		# a = []
		# if(P==0):
		# 	P = v.shape[0]
		# a = [np.rint(np.sum([v[u]*np.exp(math.pi*2j*u*k/P) for u in range(P)])) for k in range(v.shape[0])]
		a = np.fft.ifft2(np.fft.ifftshift(v),axes=(-1,))
		# a = [np.rint(np.sum([v[k-1+(P//2)+((v.shape[0]-P)//2)]*np.exp(math.pi*2j*(i+1)*k/P) for k in range(1-(P//2),1+(P//2))])) for i in range(v.shape[0])]
	else:
		# P = v.shape[0]
		# a = [np.sum([v[k]*np.exp(-2j*math.pi*(u+1)*k/P) for k in range(P)]) for u in range(P)]
		a = np.fft.fftshift(np.fft.fft2(v,axes=(-1,)))
		# a = [np.sum([v[i]*np.exp(-2j*math.pi*k*(i+1)/P) for i in range(P)])/P for k in range(1-(P//2),(P//2)+1)]
	return np.array(a)


# 3  2  1
# 4  X  0
# 5  6  7
def freeman(im):
	i,j = pbranco(im)	# search for the first white dot
	f = []
	b = [(i,j)]
	prev = -1
	while(True):
		prev = (prev + 1) % 8
		if((j == 0) and (prev >= 3) and (prev <= 5)):
			prev = 6
		elif((j == (im.shape[1] - 1)) and ((prev == 7) or (prev == 0) or (prev == 1))):
			prev = 2
		if((i == 0) and (prev >= 1) and (prev <= 3)):
			prev = 4
			if(j == 0):
				prev = 6
		elif((i == (im.shape[0] - 1)) and (prev >= 5) and (prev <= 7)):
			prev = 0
			if(j == (im.shape[1] - 1)):
				prev = 2
		tup = (i,j)
		if(prev == 0):
			tup = (i,j+1)
		elif(prev == 1):
			tup = (i-1,j+1)
		elif(prev == 2):
			tup = (i-1,j)
		elif(prev == 3):
			tup = (i-1,j-1)
		elif(prev == 4):
			tup = (i,j-1)
		elif(prev == 5):
			tup = (i+1,j-1)
		elif(prev == 6):
			tup = (i+1,j)
		elif(prev == 7):
			tup = (i+1,j+1)
		else:
			print('error:',b,prev)
			return b
		if(im[tup] >= 225):
			f.append(prev)
			if(b[0] == tup):
				return f
			elif(tup in b):
				return f
			else:
				b.append(tup)
				i,j = tup
				prev = (prev + 4) % 8
	return f


def q1(path):
	print('\nQ1')
	im = np.array(Image.open(path+'oranges.jpg'))
	hsv = rgbtohsv(im)
	H = hsv[:,:,0]
	S = hsv[:,:,1]
	I = hsv[:,:,2]
	tabela = np.logical_and(H <= 70, H > 10)
	# tabela = np.logical_and(tabela, I < 0.9)
	H[tabela] = 120
	# tabela = np.logical_and(H <= 70, H > 10)
	# tabela = np.logical_and(tabela, I > 0.9)
	hsv[:,:,0] = H
	# hsv[:,:,1] = S
	rgb = hsvtorgb(hsv)
	im = Image.fromarray(rgb)
	im.save(path+'output/q1.jpg')


def q2(path):
	print('\nQ2')
	im = np.array(Image.open(path+'Fig8.02(a).jpg'))
	res = dilatar(erodir(im))
	im = Image.fromarray(res)
	im.save(path+'output/q2a.jpg') # saving post-filter image
	k = otsu(res)
	res2 = ((res > k)*255).astype(np.uint8)
	im = Image.fromarray(res2).convert('1')
	im.save(path+'output/q2b.jpg')
	res3 = laplaciano(res2)
	im = Image.fromarray(res3).convert('1')
	im.save(path+'output/q2c.jpg')
	bordas = (res3[res3.shape[0]//2,:] > 0).sum()
	print('Number of white pixels horizontally in mid vertical:',bordas)
	print('Therefore, the number of matches is:',bordas//2)
	print('Number of white pixels:',np.unique(res2, return_counts=True)[1][1])


def q3(path):
	print('\nQ3')
	im = np.array(Image.open(path+'circulos.png').convert('L'))
	inv = 255-im
	res = np.zeros(im.shape,dtype=np.uint8)
	res[60,45] = 255
	res[180,320] = 255
	res[170,610] = 255
	res[195,960] = 255
	res[325,765] = 255
	res[430,200] = 255
	res[640,70] = 255
	res[485,390] = 255
	res[550,665] = 255
	res[625,880] = 255
	res[410,1050] = 255
	imr = Image.fromarray(im+res)
	imr.save(path+'output/q3a.jpg')
	imr = Image.fromarray(inv)
	imr.save(path+'output/q3b.jpg')
	res = dilatar(res,size=20)
	imr = Image.fromarray(res)
	imr.save(path+'output/q3c.jpg')
	res = dilatar(res,size=60)
	imr = Image.fromarray(res)
	imr.save(path+'output/q3d.jpg')
	res = ((res==inv)*255).astype(np.uint8)
	imr = Image.fromarray(res)
	imr.save(path+'output/q3e.jpg')
	res = np.maximum(res,im)
	imr = Image.fromarray(res)
	imr.save(path+'output/q3f.jpg')


def q4(path):
	print('\nQ4')
	im = np.array(Image.open(path+'estrada.jpg').convert('L'))
	k = otsu(im)
	res = ((im > k)*255).astype(np.uint8)
	imr = Image.fromarray(res)
	imr.save(path+'output/q4a.jpg')
	res2 = laplaciano(res)
	imr = Image.fromarray(res2)
	imr.save(path+'output/q4b.jpg')
	res3 = hough(res2)
	imr = Image.fromarray(res3)
	imr.save(path+'output/q4c.jpg')
	args = np.argwhere(res3 > 220)
	argsv = args[:,1] <= (4*(65 + 91))
	argsv = np.logical_or(argsv, args[:,1] >= (4*(55 + 91)))
	argsv = np.logical_or(argsv, args[:,1] >= (4*(91 - 65)))
	argsv = np.logical_or(argsv, args[:,1] <= (4*(91 - 55)))
	argsv = np.logical_or(argsv, args[:,1] >= (4*(75 + 91)))
	argsv = np.logical_or(argsv, args[:,1] <= (4*(91 - 75)))
	args = args[argsv]
	D=int(np.ceil(np.sqrt(im.shape[0]*im.shape[0]+im.shape[1]*im.shape[1])))
	for a,b in args:
		p = (a*D/250)-D
		th = (b/4)-91
		for y in range(im.shape[1]):
			x = int(np.rint((p-y*np.sin(th*math.pi/180))/np.cos(th*math.pi/180)))
			res2[x,y] = 255
	imr = Image.fromarray(res2)
	imr.save(path+'output/q4d.jpg')


def q5(path):
	print('\nQ5')
	im = np.array(Image.open(path+'Fig10.40(a).jpg').convert('L'))
	res1 = dilatar(erodir(limiarizar(im,240),size=3),size=3)
	imr = Image.fromarray(res1)
	imr.save(path+'output/q5a.jpg')
	res2 = erodir(res1,size=3)
	imr = Image.fromarray(res2)
	imr.save(path+'output/q5b.jpg')
	seeds = erodir(res2,size=3,dot=True)
	for i in range(10):
		seeds = erodir(seeds,size=3,dot=True)
	imr = Image.fromarray(seeds)
	imr.save(path+'output/q5c.jpg')
	res4 = np.abs(255-im.astype(np.int))
	imr = Image.fromarray(res4.astype(np.uint8))
	imr.save(path+'output/q5d.jpg')
	res5 = limiarizar(res4,68)
	imr = Image.fromarray(res5)
	imr.save(path+'output/q5e.jpg')
	res6 = crescer(seeds,res5)
	imr = Image.fromarray(res6)
	imr.save(path+'output/q5f.jpg')


def q6(path):
	print('\nQ6')
	img = Image.open(path+'lena.tif')
	im = np.array(img)
	print('a)')
	# moments(im)
	print('\nb)')
	im = np.array(img.resize((im.shape[0]//2,im.shape[1]//2)))
	img.resize((im.shape[0]//2,im.shape[1]//2)).save(path+'output/q6b.jpg')
	moments(im)
	print('\nc)')
	im = np.array(img.rotate(90))
	img.rotate(90).save(path+'output/q6c.jpg')
	moments(im)
	print('\nd)')
	im = np.array(img.rotate(180))
	img.rotate(180).save(path+'output/q6d.jpg')
	moments(im)


def q7(path):
	print('\nQ7')
	img = Image.open('Fig11.10.jpg').convert('L')
	im = np.array(img)
	k = otsu(im)
	im = ((im > k)*255).astype(np.uint8)
	res = laplaciano(im)
	Image.fromarray(res).save(path+'output/q7a.jpg')
	border = fronteira(res)	# obter a fronteira (lista de pixels - x,y)
	sk = complexar(border)	# obter s(k) = x(k) + jy(k)
	au = descritores(sk)	# obter descritores - transformada
	aux = np.array(au)	# copiar vetor
	# ~10%
	aux[:(aux.shape[0]//2)-50] = 0
	aux[(aux.shape[0]//2)+50:] = 0
	skx = descritores(aux,inv=True)
	res = descomplexar(skx)
	imx = np.array(Image.new('L',img.size), dtype=np.uint8)
	for i,j in res:
		imx[i,j] = 255
	Image.fromarray(imx).save(path+'output/q7b.jpg')
	# ~5%
	aux[:(aux.shape[0]//2)-25] = 0
	aux[(aux.shape[0]//2)+25:] = 0
	skx = descritores(aux,inv=True)
	res = descomplexar(skx)
	imx = np.array(Image.new('L',img.size), dtype=np.uint8)
	for i,j in res:
		imx[i,j] = 255
	Image.fromarray(imx).save(path+'output/q7c.jpg')
	# ~2.5%
	aux[:(aux.shape[0]//2)-12] = 0
	aux[(aux.shape[0]//2)+12:] = 0
	skx = descritores(aux,inv=True)
	res = descomplexar(skx)
	imx = np.array(Image.new('L',img.size), dtype=np.uint8)
	for i,j in res:
		imx[i,j] = 255
	Image.fromarray(imx).save(path+'output/q7d.jpg')
	# ~1.25%
	aux[:(aux.shape[0]//2)-6] = 0
	aux[(aux.shape[0]//2)+6:] = 0
	skx = descritores(aux,inv=True)
	res = descomplexar(skx)
	imx = np.array(Image.new('L',img.size), dtype=np.uint8)
	for i,j in res:
		imx[i,j] = 255
	Image.fromarray(imx).save(path+'output/q7e.jpg')
	# ~0.63%
	aux[:(aux.shape[0]//2)-3] = 0
	aux[(aux.shape[0]//2)+3:] = 0
	skx = descritores(aux,inv=True)
	res = descomplexar(skx)
	imx = np.array(Image.new('L',img.size), dtype=np.uint8)
	for i,j in res:
		imx[i,j] = 255
	Image.fromarray(imx).save(path+'output/q7f.jpg')
	# ~0.28%
	aux[:(aux.shape[0]//2)-1] = 0
	aux[(aux.shape[0]//2)+1:] = 0
	skx = descritores(aux,inv=True)
	res = descomplexar(skx)
	imx = np.array(Image.new('L',img.size), dtype=np.uint8)
	for i,j in res:
		imx[i,j] = 255
	Image.fromarray(imx).save(path+'output/q7g.jpg')


def q8(path):
	print('\nQ8')
	img = Image.open('Fig11.10.jpg').convert('L').resize((9,15),Image.NEAREST)
	im = np.array(img)
	k = otsu(im)
	res1 = ((im > k)*255).astype(np.uint8)
	# Image.fromarray(res1).resize((30,50),resample=Image.NEAREST).show()
	res2 = laplaciano(res1)
	img = Image.fromarray(res2)
	img.save(path+'output/q8a.jpg')
	res3 = freeman(res2)
	print("Freeman's sequence:",res3)


def main(path='/Users/mthome/Dropbox/UFES/Processamento Digital de Imagens/2lista/'):
	try:
		os.mkdir(path+'output/')
	except FileExistsError as e:
		pass
	except:
		print('error while creating output dir')
	# q1(path)
	# q2(path)
	# q3(path)
	# q4(path)
	# q5(path)
	# q6(path)
	# q7(path)
	q8(path)

if(__name__ == '__main__'):
	main()

# Q6
# a)
# 1st: 0.0013361711781900852
# 2nd: 6.793361954713708e-09
# 3rd: 1.1930056615324684e-12
# 4th: 7.230356065403246e-12
# 5th: -1.8366333356640672e-24
# 6th: -8.796692348152943e-16
# 7th: -4.27084914254567e-23
#
# b)
# 1st: 0.001335502908569982
# 2nd: 6.6486184681755276e-09
# 3rd: 1.070747475917829e-12
# 4th: 7.1000899006459604e-12
# 5th: -3.8326817865674265e-25
# 6th: -8.693685589281759e-16
# 7th: -4.063889981068169e-23
#
# c)
# 1st: 0.0013361711781900852
# 2nd: 6.793361954713708e-09
# 3rd: 1.1930056615324684e-12
# 4th: 8.948157891619557e-12
# 5th: -1.8366333356640672e-24
# 6th: -8.796692348152943e-16
# 7th: -4.27084914254567e-23
#
# d)
# 1st: 0.0013361711781900852
# 2nd: 6.793361954713708e-09
# 3rd: 1.1930056615324684e-12
# 4th: 7.230356065403246e-12
# 5th: -1.8366333356640672e-24
# 6th: -8.796692348152943e-16
# 7th: -4.27084914254567e-23
