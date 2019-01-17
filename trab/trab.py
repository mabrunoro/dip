#!/usr/bin/env python3

import numpy as np
from PIL import Image
import math
import sys
import os

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
				ang = np.arctan(dy*180/(dx*math.pi))
				inters = (dx,dy,ang)
				# print(ang)
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
	# for i in range(10,im.shape[0]-10):
	# 	for j in range(10,im.shape[1]-10):
	crosses = []
	for i in range(3750,4750):
		for j in range(5750,6500):
			if(im[i,j] < 200):
				t = checkcross(im,i,j)
				if(t is not None):
					res[i+t[0]-10:i+t[0]+10,j+t[1]-10:j+t[1]+10] = np.uint8(255)
					crosses.append((i+t[0],j+t[1]))
					break
					# res[i-10:i+10,j-10:j+10] = np.uint8(150)
					# return res,t[2]

		for j in range(400,1150):
			if(im[i,j] < 200):
				t = checkcross(im,i,j)
				if(t is not None):
					res[i+t[0]-10:i+t[0]+10,j+t[1]-10:j+t[1]+10] = np.uint8(255)
					crosses.append((i+t[0],j+t[1]))
					break
					# res[i-10:i+10,j-10:j+10] = np.uint8(150)
					# return res,t[2]

	for i in range(1500,2000):
		for j in range(5750,6500):
			if(im[i,j] < 200):
				# k=0
				# while(im[i,j+k] < 200):
				# 	k += 1
				# t = checkcross(im,i,j+k//2)
				t = checkcross(im,i,j)
				if(t is not None):
					res[i+t[0]-10:i+t[0]+10,j+t[1]-10:j+t[1]+10] = np.uint8(255)
					crosses.append((i+t[0],j+t[1]))
					break
					# res[i-10:i+10,j-10:j+10] = np.uint8(150)
					# return res,t[2]

		for j in range(400,1150):
			if(im[i,j] < 200):
				t = checkcross(im,i,j)
				if(t is not None):
					res[i+t[0]-10:i+t[0]+10,j+t[1]-10:j+t[1]+10] = np.uint8(255)
					crosses.append((i+t[0],j+t[1]))
					break
					# res[i-10:i+10,j-10:j+10] = np.uint8(150)
					# return res,t[2]
	return res,crosses

def prin(img):
	img = img.convert('L')
	# img.show()
	im = np.array(img)
	k = otsu(im)
	res1 = limiarizar(im,k)
	# for i in range(im.shape[0]//2):
	# 	res1[i,2*i] = 0
	# Image.fromarray(res1).show()
	# res2 = dilatar(erodir(im,size=3),size=3)
	# Image.fromarray(res2).show()

	# res1[]

	res3,crosses = lookcross(im)
	sumx = 0
	sumy = 0
	for i,j in crosses:
		sumx += i
		sumy += j
	sumx = sumx // 4
	sumy = sumy // 4
	print(crosses,sumx,sumy)
	res3[sumx-5:sumx+5,sumy-5:sumy+5] = np.uint8(255)
	Image.fromarray(res3).show()
	# img.rotate(-ang).show()

	# res2 = laplaciano(res)
	# Image.fromarray(res2).show()


def main(path='/Users/mthome/Dropbox/UFES/Processamento Digital de Imagens/trab/'):
	try:
		os.mkdir(path+'output/')
	except FileExistsError as e:
		pass
	except:
		print('error while creating output dir')
	# orig = pngtorgb(Image.open(path+'original.png'))
	# print(pca(np.array(orig)))
	# im = pngtorgb(Image.open(path+'G0.png'))
	# print(pca(np.array(im)))
	# im = Image.open(path+'G1.png')
	# print(pca(np.array(im)))
	im = Image.open(path+'G2.png')
	print(pca(np.array(im)))
	prin(im)

	# im.rotate(ang*180/math.pi).show()

	# prin(im)

if(__name__ == '__main__'):
	main()
