#!/usr/bin/env python3

import numpy as np
from PIL import Image
import math
import sys
import os

MAXIMOS = [1,5,10,50,100,500,1000]

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
	if(im.shape == (5100, 7014)):
		crosses = [slicecross(im,3750,4750,5750,6500), slicecross(im,3750,4750,400,1150), slicecross(im,1500,2000,5750,6500), slicecross(im,1500,2000,400,1150)]
	else:
		crosses = [slicecross(im,3500,4000,5300,5850), slicecross(im,3500,4000,300,800), slicecross(im,1250,1600,5300,5850), slicecross(im,1250,1600,300,800)]
	return crosses

def checkcolor(img):
	clrs = img.mean(axis=1).mean(axis=0)
	if(np.any(clrs < 250)):
		return clrs
	else:
		return None

def limpar(img,size=13):
	for i in range(size,img.shape[0]-size):
		for j in range(size,img.shape[1]-size):
			if(img[i,j] and not (np.any(img[i-size:i+size,j-size]) or np.any(img[i-size:i+size,j+size]) or np.any(img[i-size,j-size:j+size]) or np.any(img[i+size,j-size:j+size]))):
				img[i,j] = False
	return img

def prin(img,ori,pontos=30):
	im = np.array(img.convert('L'))
	k = otsu(im)

	crossimg = lookcross(limiarizar(im,k))	# busca as cruzes na imagem (sem rotacionar)

	angimg = np.arctan((crossimg[0][0]-crossimg[1][0])/(crossimg[0][1]-crossimg[1][1]))*180/math.pi	# descobre o angulo em relação à original
	imgrot = img.rotate(angimg)	# rotaciona a imagem colorida
	res1 = np.array(imgrot)	# array da imagem colorida

	res2 = np.array(imgrot.convert('L'))	# img rotacionada e em escala de cinza - array
	crossimg = lookcross(limiarizar(res2,k))	# obtém as cruzes na imagem rotacionada
	imgcross = Image.fromarray(res1[crossimg[3][0]:crossimg[0][0],crossimg[3][1]:crossimg[0][1],:])	# separa o plano dos eixos da imagem colorida rotacionada

	orig = ori.convert('L')	# converte imagem original pra escala de cinza (começa como RGB)
	res3 = limiarizar(np.array(orig),200)	# imagem original binarizada
	crossori = lookcross(res3)	# procura as cruzes na imagem original
	# res4 = np.uint8(255)-res3[crossori[3][0]:crossori[0][0],crossori[3][1]:crossori[0][1]]
	res4 = dilatar(np.uint8(255)-res3[crossori[3][0]:crossori[0][0],crossori[3][1]:crossori[0][1]],3)
	res5 = Image.fromarray(res4)

	res6 = imgcross.resize(res5.size)	# iguala o tamanho da imagem ao da imagem original
	# res5.show()
	# res6.show()

	res7 = np.array(res6,dtype=np.int16) + np.array(res5.convert('RGB'),dtype=np.int16)	# faz a diferença entre o plano da imagem e o da imagem original (remover eixos)
	res8 = np.array(res7.clip(min=0,max=255),dtype=np.uint8)	# padroniza valores, removendo os negativos
	# Image.fromarray(res8).show()
	res9 = afinar(res8,size=3)	# limpa a image resultante
	# Image.fromarray(res9).show()

	# obtém os dados das curvas: cores, máximos e escalas
	curvas = [[None,-1,None,-1,None],[None,-1,None,-1,None],[None,-1,None,-1,None]]
	if(ori.size != img.size):
		rx = (crossimg[3][0]-crossimg[1][0])/(crossori[3][0]-crossori[1][0])
		ry = (crossimg[3][1]-crossimg[2][1])/(crossori[3][1]-crossori[2][1])
		for j in range(3):
			# X
			for i in range(7):
				col = checkcolor(res1[crossimg[3][0]+round((j*188-752)*rx):crossimg[3][0]+round((j*188-693)*rx),crossimg[3][1]+round((i*256-503)*ry):crossimg[3][1]+round((i*256-440)*ry)])
				if(col is not None):
					curvas[j][0] = col
					curvas[j][1] = i
					break
			col = checkcolor(res1[crossimg[3][0]+round((j*188-752)*rx):crossimg[3][0]+round((j*188-693)*rx),crossimg[3][1]+round(1451*ry):crossimg[3][1]+round(1514*ry)])
			if(col is None):
				col = checkcolor(res1[crossimg[3][0]+round((j*188-752)*rx):crossimg[3][0]+round((j*188-693)*rx),crossimg[3][1]+round(2008*ry):crossimg[3][1]+round(2071*ry)])
				curvas[j][2] = True	# logaritmo
			else:
				curvas[j][2] = False	# linear

			# Y
			for i in range(7):
				col = checkcolor(res1[crossimg[3][0]+round((j*188-752)*rx):crossimg[3][0]+round((j*188-693)*rx),crossimg[3][1]+round((i*256+2945)*ry):crossimg[3][1]+round((i*256+3008)*ry)])
				if(col is not None):
					# curvas[j][0] = col
					curvas[j][3] = i
					break
			col = checkcolor(res1[crossimg[3][0]+round((j*188-752)*rx):crossimg[3][0]+round((j*188-693)*rx),crossimg[3][1]+round(4899*ry):crossimg[3][1]+round(4962*ry)])
			if(col is None):
				col = checkcolor(res1[crossimg[3][0]+round((j*188-752)*rx):crossimg[3][0]+round((j*188-693)*rx),crossimg[3][1]+round(5456*ry):crossimg[3][1]+round(5519*ry)])
				curvas[j][4] = True	# logaritmo
			else:
				curvas[j][4] = False	# linear
	else:
		for j in range(3):
			for i in range(7):
				col = checkcolor(res1[crossori[3][0]-752+j*188:crossori[3][0]-693+j*188,crossori[3][1]-503+i*256:crossori[3][1]-440+i*256])
				if(col is not None):
					curvas[j][0] = col
					curvas[j][1] = i
					break
			col = checkcolor(res1[crossori[3][0]-752+j*188:crossori[3][0]-693+j*188,crossori[3][1]+1451:crossori[3][1]+1514])
			if(col is None):
				col = checkcolor(res1[crossori[3][0]-752+j*188:crossori[3][0]-693+j*188,crossori[3][1]+2008:crossori[3][1]+2071])
				curvas[j][2] = True	# logaritmo
			else:
				curvas[j][2] = False	# linear
			for i in range(7):
				col = checkcolor(res1[crossori[3][0]-752+j*188:crossori[3][0]-693+j*188,crossori[3][1]+2945+i*256:crossori[3][1]+3008+i*256])
				if(col is not None):
					# curvas[j][0] = col
					curvas[j][3] = i
					break
			col = checkcolor(res1[crossori[3][0]-752+j*188:crossori[3][0]-693+j*188,crossori[3][1]+4899:crossori[3][1]+4962])
			if(col is None):
				col = checkcolor(res1[crossori[3][0]-752+j*188:crossori[3][0]-693+j*188,crossori[3][1]+5456:crossori[3][1]+5519])
				curvas[j][4] = True	# logaritmo
			else:
				curvas[j][4] = False	# linear

	for i in range(3):
		print('Curva',i+1)
		print('\tRGB:',curvas[i][0][0],curvas[i][0][1],curvas[i][0][2])
		print('\tXmax:',MAXIMOS[curvas[i][1]])
		print('\tEscala X:','log' if curvas[i][2] else 'linear')
		print('\tYmax:',MAXIMOS[curvas[i][3]])
		print('\tEscala Y:','log' if curvas[i][4] else 'linear')
	# Image.fromarray(res1).show()
	# res10 = (limpar(np.any(res9 < np.array([250,250,250]),axis=2))*np.uint8(255)).astype(np.uint8)
	# Image.fromarray(res10).show()
	res10 = np.argwhere(limpar(np.any(res9 < np.array([250,250,250]),axis=2)))
	# res11 = np.zeros(res9.shape,dtype=np.uint8)
	# for i,j in res10:
	# 	res11[i,j,:] = res9[i,j,:]
	# Image.fromarray(res9).show()
	# Image.fromarray(res11).show()
	if(curvas[0][0] is not None):
		if(curvas[1][0] is not None):
			if(curvas[2][0] is not None):
				grupos = [[],[],[]]
			else:
				grupos = [[],[]]
		else:
			grupos = [[]]
	else:
		print('Não há curvas identificadas')
		sys.exit(0)
	centroX = res6.size[0]
	centroY = res6.size[1]
	# for i,j in crossori:
	# 	centroX += i
	# 	centroY += j
	# centroX = centroX // 4
	# centroY = centroY // 4
	# DY = 1042
	# DX = 2084
	aux6 = np.array(res6)
	for i,j in res10:
		dists = np.abs([np.linalg.norm(curvas[k][0].astype(np.int16) - aux6[i,j,:].astype(np.int16)) for k in range(len(grupos))])
		corp = np.argpartition(dists,(0,1))
		if(np.abs(dists[corp[0]]-dists[corp[1]]) < 50):
			varponto = np.var(aux6[i,j,:])
			var0 = np.var(curvas[corp[0]][0])
			var1 = np.var(curvas[corp[1]][0])
			cor = corp[0] if (np.abs(varponto - var0) < np.abs(varponto - var1)) else corp[1]
		else:
			cor = corp[0]
		X = ('10^'+str((j-centroY)*MAXIMOS[curvas[cor][1]]/2084)) if curvas[cor][2] else str((j-centroY)*MAXIMOS[curvas[cor][1]]/2084)
		Y = ('10^'+str((centroX-i)*MAXIMOS[curvas[cor][3]]/1042)) if curvas[cor][4] else str((centroX-i)*MAXIMOS[curvas[cor][3]]/1042)
		grupos[cor].append([X,Y])	# X e Y são trocados no numpy

	for i in range(len(grupos)):
		with open('output/curva'+str(i)+'.txt','w') as f:
			f.write('Cor RGB: '+str(curvas[i][0]))
			# print(len(grupos[i]))
			passo = (len(grupos[i])//pontos) if (pontos < len(grupos[i])) else 1
			for j in range(0,len(grupos[i]),passo):
				f.write(str(grupos[i][j]))


def main(pontos=30,path='/Users/mthome/Dropbox/UFES/Processamento Digital de Imagens/trab/'):
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
	prin(im,orig,pontos)


if(__name__ == '__main__'):
	if(len(sys.argv) > 1):
		main(int(sys.argv[1]))
	else:
		main()
