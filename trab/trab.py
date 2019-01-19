#!/usr/bin/env python3

import numpy as np
from PIL import Image
import math
import sys
import os

def erodir(im,size=5,dot=False):
	res = np.zeros(im.shape, dtype=np.uint8)
	side = size//2 if size > 1 else 1
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

def dilatar(im,size=5):
	res = np.zeros(im.shape, dtype=np.uint8)
	side = size//2 if size > 1 else 1
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			sti = i - side if i >= side else 0
			eni = i + side if i < (im.shape[0]-side) else im.shape[0]
			stj = j - side if j >= side else 0
			enj = j + side if j < (im.shape[1]-side) else im.shape[1]
			res[i,j] = np.max(im[sti:eni,stj:enj])
	return res

def afinar(im,size=2):
	res = np.ones(im.shape, dtype=np.uint8)*np.uint8(255)
	aux = im < 250
	for i in range(size,im.shape[0]-size):
		for j in range(size,im.shape[1]-size):
			if(np.all(np.any(aux[i-size:i+size+1,j-size:j+size+1,:],axis=2))):
				res[i,j,:] = im[i-size:i+size+1,j-size:j+size+1,:].mean(axis=1).mean(axis=0)
	return res

def P12(l,n,MN,k):
	p1 = np.sum(n[l <= k])/MN
	return (p1,1-p1)

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

def limiarizar(im,val,inv=False):
	if(inv):
		return ((im < val)*255).astype(np.uint8)
	else:
		return ((im > val)*255).astype(np.uint8)

def pngtorgb(im):
	b = Image.new('RGB', im.size, (255,255,255))
	b.paste(im, mask=im.split()[3])
	return b

def pca(im):
	img = np.array(im)
	coords = np.argwhere(img < 255)
	mx = np.mean(coords,axis=0)
	cx = np.cov(coords.T)
	w,v = np.linalg.eig(cx)
	A = -v.T
	# lower value signs lower angle with horizontal axis
	if(w[0] > w[1]):
		ang = np.arctan(v[0,0]/v[1,0])
	else:
		ang = np.arctan(v[0,1]/v[1,1])
	return ang*180/math.pi

def checkcross(im,x,y,limiar=200):
	dx = 0
	dy = 0
	inters = None
	while(True):
		dx += 1
		if(x + dx >= im.shape[0]):
			return inters
		elif(im[x+dx,y+dy] > limiar):
			# k = 0
			# while((y+dy+k < im.shape[1]) and (im[x+dx,y+dy+k] < limiar) and ())
			if(im[x+dx,y+dy+1] < limiar):
				dy += 1
			elif(im[x+dx,y+dy+2] < limiar):
				dy += 2
			elif(im[x+dx,y+dy-1] < limiar):
				dy -= 1
			elif(im[x+dx,y+dy-2] < limiar):
				dy -= 2
			else:
				return inters
		elif(dx > 20):
			if((im[x+dx-(dy//2),y+dy+(dx//2)] < limiar) and (im[x+dx-(dy//2),y+dy-(dx//2)] < limiar)):
				inters = (dx,dy)
				return inters

def slicecross(im,i1,i2,j1,j2,res=None):
	for i in range(i1,i2):
		for j in range(j1,j2):
			if(im[i,j] < 200):
				t = checkcross(im,i,j)
				if(t is not None):
					if(res is not None):
						res[i+t[0]-10:i+t[0]+10,j+t[1]-10:j+t[1]+10] = np.uint8(255)
					return (i+t[0],j+t[1])

def lookcross(im):
	res = np.zeros(im.shape, dtype=np.uint8)
	# 2181-1062
	if(im.shape == (5100, 7014)):
		crosses = [slicecross(im,3750,4750,5750,6500,res), slicecross(im,3750,4750,400,1150,res), slicecross(im,1500,2000,5750,6500,res), slicecross(im,1500,2000,400,1150,res)]
	else:
		crosses = [slicecross(im,3500,4000,5300,5850,res), slicecross(im,3500,4000,300,800,res), slicecross(im,1250,1600,5300,5850,res), slicecross(im,1250,1600,300,800,res)]
	return res,crosses

def prin(img,ori):
	im = np.array(img.convert('L'))
	k = otsu(im)
	res1 = limiarizar(im,k)

	orig = ori.convert('L')

	# for i in range(im.shape[0]//2):
	# 	res1[i,2*i] = 0
	# Image.fromarray(res1).show()
	# res2 = dilatar(erodir(im,size=3),size=3)
	# Image.fromarray(res2).show()

	# res1[]

	res2,crosses = lookcross(im)
	sumx = 0
	sumy = 0
	for i,j in crosses:
		sumx += i
		sumy += j
	sumx = sumx // 4
	sumy = sumy // 4
	# print(crosses,sumx,sumy,suma)
	# res2[sumx-5:sumx+5,sumy-5:sumy+5] = np.uint8(255)
	# Image.fromarray(res2).show()
	# img.rotate(-suma).show()

	ang = np.arctan((crosses[0][0]-crosses[1][0])/(crosses[0][1]-crosses[1][1]))*180/math.pi
	print(ang)

	im3 = img.rotate(ang)
	res3 = np.array(im3.convert('L'))
	# res3[sumx-5:sumx+5,sumy-5:sumy+5,0] = np.uint8(255)
	# Image.fromarray(res3).show()
	_,crosses = lookcross(res3)
	res4 = Image.fromarray(np.array(im3)[crosses[3][0]:crosses[0][0],crosses[3][1]:crosses[0][1],:])

	res5 = limiarizar(np.array(orig),200)
	_,crosses = lookcross(res5)
	res6 = dilatar(np.uint8(255)-res5[crosses[3][0]:crosses[0][0],crosses[3][1]:crosses[0][1]],3)
	res6 = Image.fromarray(res6)

	res7 = res4.resize(res6.size)
	# res7.show()
	# res6.show()

	res8 = np.array(res7,dtype=np.int16) + np.array(res6.convert('RGB'),dtype=np.int16)
	res9 = np.array(res8.clip(min=0,max=255),dtype=np.uint8)
	Image.fromarray(res9).show()

	res10 = afinar(res9,size=3)
	Image.fromarray(res10).show()


def main(path='/Users/mthome/Dropbox/UFES/Processamento Digital de Imagens/trab/'):
	try:
		os.mkdir(path+'output/')
	except FileExistsError as e:
		pass
	except:
		print('error while creating output dir')
	orig = pngtorgb(Image.open(path+'original.png'))
	# im = pngtorgb(Image.open(path+'G0.png'))
	# im = Image.open(path+'G1.png')
	im = Image.open(path+'G2.png')
	prin(im,orig)

	# im.rotate(ang*180/math.pi).show()

	# prin(im)

if(__name__ == '__main__'):
	main()
