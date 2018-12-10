#!/usr/bin/env python3
import os
import sys
import numpy as np
from PIL import Image
import skimage.color as ski
import matplotlib.pyplot as plt


# hist: lista contendo a quantidade de pixels com a intensidade indicada pelo index
def ehist(hist):
	hist = np.array(hist)
	NM = hist.sum()
	s = np.zeros(hist.size, dtype=int)
	for i in range(hist.size):
		s[i] = round((hist.size-1)*hist[:i+1].sum()/NM)
	return s


# define um histograma para a imagem
def dhist(he, prob):
	he = np.array(he)
	gz = np.zeros(he.size, dtype=int)
	for i in range(he.size):
		gz[i] = round(prob[:i+1].sum()*(he.size-1))

	mp = np.zeros(he.size, dtype=int)
	for i in np.unique(he):
		mp[i] = np.argmin(np.abs(gz-i))
	return mp


# imprime a média e variância amostrais da imagem
def imv(im):
	d = np.array([i for i in im.tobytes()])
	print('\tMédia amostral:',d.mean())
	print('\tVariância amostral:',d.var())


# Questão 1
def q1(path=''):
	# Fig3.15(a)
	print('Questão 1')
	print('Fig3.15')
	fig315 = Image.open(path+'Fig3.15(a).jpg')
	h = fig315.histogram()
	plt.figure('Fig3.15(a) Original Histogram')
	plt.plot(h)
	print('Original')
	imv(fig315)
	s1 = ehist(h)

	# histograma desejado na segunda parte
	prob = np.zeros(s1.size)
	prob[:120] = 1/(256-80)
	prob[200:] = 1/(256-80)

	equalizado = [int(s1[i]) for i in fig315.tobytes()]
	fig315.putdata(equalizado)
	fig315.save(path+'output/out1_fig315_eq.jpg')
	plt.figure('Fig3.15(a) Equalized Histogram')
	plt.plot(fig315.histogram())
	print('Histograma Equalizado')
	imv(fig315)

	s2 = dhist(s1, prob)
	customizado = [int(s2[i]) for i in equalizado]
	fig315.putdata(customizado)
	fig315.save(path+'output/out1_fig315_ct.jpg')
	plt.figure('Fig3.15(a) Custom Histogram')
	plt.plot(fig315.histogram())
	plt.show()
	print('Histograma Personalizado')
	imv(fig315)

	# train
	print('\nTrain')
	train = Image.open(path+'train.jpg')
	h = train.histogram()
	plt.figure('Train Original Histogram')
	plt.plot(h)
	print('Original')
	imv(train)

	s1 = ehist(h)
	equalizado = [int(s1[i]) for i in train.tobytes()]
	train.putdata(equalizado)
	train.save(path+'output/out1_train_eq.jpg')
	plt.figure('Train Equalized Histogram')
	plt.plot(train.histogram())
	print('Histograma Equalizado')
	imv(train)

	s2 = dhist(s1, prob)
	customizado = [int(s2[i]) for i in equalizado]
	train.putdata(customizado)
	train.save(path+'output/out1_train_ct.jpg')
	plt.figure('Train Custom Histogram')
	plt.plot(train.histogram())
	plt.show()
	print('Histograma Personalizado')
	imv(train)


# calcula a correlação entre as imagens (fornecer os objetos das imagens)
def corri(tp, im):
	template = np.array(tp).T
	tmean = template.mean()
	tmmean = (template - tmean).flatten()
	tstd = template.std()
	# image = np.zeros((im.size[0]+2*tp.size[0]-2,im.size[1]+2*tp.size[1]-2))
	# image[tp.size[0]-1:1-tp.size[0],tp.size[1]-1:1-tp.size[1]] = np.array(im).T
	image = np.array(im).T
	# corr = np.zeros((im.size[0]+tp.size[0]-1,im.size[1]+tp.size[1]-1))
	corr = np.zeros(im.size)
	for i in range(im.size[0]-tp.size[0]):
		for j in range(im.size[1]-tp.size[1]):
			aux = image[i:i+tp.size[0],j:j+tp.size[1]].flatten()
			corr[i,j] = np.dot(tmmean,(aux - aux.mean()))/(template.size*tstd*aux.std())
	return corr


# Questão 2
def q2(path):
	print('Questão 2')
	template = Image.open(path+'template.jpg').convert('L')
	puzzle = Image.open(path+'puzzle_1.jpg').convert('L')
	res = corri(template, puzzle)
	cormax = res.argmax()
	cormax = (cormax//puzzle.size[1], cormax%puzzle.size[1])
	npuzzle = np.ones(puzzle.size)*0.5
	npuzzle[cormax[0]:cormax[0]+template.size[0],cormax[1]:cormax[1]+template.size[1]] = 1.5
	npuzzle = npuzzle.T
	npuzzle *= np.array(puzzle)
	puzzle.putdata(npuzzle.flatten())
	puzzle.save(path+'output/out2_puzzle1.jpg')

	puzzle = Image.open(path+'puzzle_2.jpg').convert('L')
	res = corri(template, puzzle)
	cormax = res.argmax()
	cormax = (cormax//puzzle.size[1], cormax%puzzle.size[1])
	npuzzle = np.ones(puzzle.size)*0.5
	npuzzle[cormax[0]:cormax[0]+template.size[0],cormax[1]:cormax[1]+template.size[1]] = 1.5
	npuzzle = npuzzle.T
	npuzzle *= np.array(puzzle)
	puzzle.putdata(npuzzle.flatten())
	puzzle.save(path+'output/out2_puzzle2.jpg')


# máscara de convolução
def mconv(mk, im, mk2=None, pb=False, extend=False, mirror=False):
	N = mk.shape[0]
	image = np.zeros((im.size[0]+2*N-2,im.size[1]+2*N-2))
	image[N-1:1-N,N-1:1-N] = np.array(im).T
	if(mk2 is not None):
		mkf2 = mk2.flatten()
		mkf = mk.flatten()
	else:
		mkf = np.flip(mk, (0,1)).flatten()	# gira 180
	conv = np.zeros((im.size[0]+N-1,im.size[1]+N-1))
	if(extend):
		image[N-1:1-N,:N-1] = np.repeat(image[N-1:1-N,N-1],N-1).reshape((im.size[0],N-1))
		image[N-1:1-N,1-N:] = np.repeat(image[N-1:1-N,1-N],N-1).reshape((im.size[0],N-1))
		image[:N-1,N-1:1-N] = np.rot90(np.repeat(image[N-1,N-1:1-N],N-1).reshape((im.size[0],N-1)))
		image[1-N:,N-1:1-N] = np.rot90(np.repeat(image[1-N,N-1:1-N],N-1).reshape((im.size[0],N-1)))
	elif(mirror):
		image[N-1:1-N,:N-1] = np.fliplr(image[N-1:1-N,N-1:2*(N-1)])
		image[N-1:1-N,1-N:] = np.fliplr(image[N-1:1-N,2*(1-N):1-N])
		image[:N-1,N-1:1-N] = np.flipud(image[N-1:2*(N-1),N-1:1-N])
		image[1-N:,N-1:1-N] = np.flipud(image[2*(1-N):1-N,N-1:1-N])
	for i in range(im.size[0]+N-1):
		for j in range(im.size[1]+N-1):
			if(pb):
				conv[i,j] = round(image[i:i+N, j:j+N].mean())
			else:
				aux = np.dot(mkf,image[i:i+N, j:j+N].flatten())
				if(mk2 is not None):
					aux += np.dot(mkf2,image[i:i+N, j:j+N].flatten())
				conv[i,j] = int(aux)
	return conv[(N-1)//2:(1-N)//2,(N-1)//2:(1-N)//2]


# Questão 3
def q3(path):
	print('Questão 3')
	lena = Image.open(path+'lena.tif')
	passab = mconv(np.ones((9,9)), lena, pb=True, mirror=True)
	oplaplac = np.array([1,1,1,1,-8,1,1,1,1]).reshape((3,3))
	laplac = mconv(oplaplac, lena, mirror=True)
	opsobel1 = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1]).reshape((3,3))
	opsobel2 = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape((3,3))
	sobel = mconv(opsobel1, lena, mk2=opsobel2, mirror=True)
	lena.putdata(passab.flatten())
	lena.save(path+'output/out3_lena_pb.tif')
	lena.putdata(laplac.flatten())
	lena.save(path+'output/out3_lena_lp.tif')
	lena.putdata(sobel.flatten())
	lena.save(path+'output/out3_lena_sb.tif')


# Questão 4
def q4(path):
	print('Questão 4')
	fig346 = Image.open(path+'Fig3.46(a).jpg')
	original = np.array(fig346).T
	oplaplac = np.array([1,1,1,1,-8,1,1,1,1]).reshape((3,3))
	laplac = np.abs(mconv(oplaplac, fig346))
	opsobel1 = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1]).reshape((3,3))
	opsobel2 = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape((3,3))
	sobel = np.abs(mconv(opsobel1, fig346, mk2=opsobel2, mirror=True))
	fig346.putdata(laplac.T.flatten())
	fig346.save(path+'output/out4_fig346_b.jpg')
	olaplac = original.T.flatten()+laplac.T.flatten()	# original + laplaciano
	fig346.putdata(olaplac.T)
	fig346.save(path+'output/out4_fig346_c.jpg')
	fig346.putdata(sobel.T.flatten())
	fig346.save(path+'output/out4_fig346_d.jpg')
	sobelmean = np.abs(mconv(np.ones((5,5)), fig346, pb=True, mirror=True))	# fig346 == resultado Sobel
	fig346.putdata(sobelmean.T.flatten())
	fig346.save(path+'output/out4_fig346_e.jpg')
	olsm = olaplac.flatten()*sobelmean.T.flatten()	# (original + laplaciano) * (sobel suavizado)
	fig346.putdata(olsm)
	fig346.save(path+'output/out4_fig346_f.jpg')
	olsm += original.T.flatten()	# original realçada
	fig346.putdata(olsm)
	fig346.save(path+'output/out4_fig346_g.jpg')
	# transformação de potência: gama=0.5 e c=1
	olsmfp = np.float_power(olsm, 0.5)
	fig346.putdata(olsmfp)
	fig346.save(path+'output/out4_fig346_h.jpg')


# convert from polar to rectangular complex (rad)
def p2c(mag,ang):
	return mag*np.exp(1j*ang)

# transformada de Fourier
def ftransform(im):
	# res = np.zeros(im.size,complex)
	# M = im.size[0]
	# N = im.size[1]
	# y = np.repeat(np.arange(0,M,1,dtype=int),N)
	# x = np.rot90(np.repeat(np.arange(0,N,1,dtype=int),M).reshape((N,M))).flatten()
	# # print(x)
	# # print(y)
	# f = np.array([i for i in im.tobytes()],dtype=complex)
	# for u in np.arange(0,M,1,dtype=int):
	# 	for v in np.arange(0,N,1,dtype=int):
	# 		res[u,v] = (f*np.exp(-2j*np.pi*((u*x/M)+(v*y/N)))).sum()
	res = np.fft.fftshift(np.fft.fft2(np.array([i for i in im.tobytes()]).reshape(im.size)))
	return (np.abs(res),np.angle(res),res)


# Questão 5
def q5(path):
	print('Questão 5')
	lena = Image.open(path+'lena.tif')
	(mags,rads,_) = ftransform(lena)
	plt.imshow(rads, cmap='Greys_r')
	plt.show()
	rads *= 1.75
	com = p2c(mags,rads)
	aux = np.abs(np.fft.ifft2(np.fft.ifftshift(com)))
	plt.imshow(aux, cmap='Greys_r')
	plt.show()

	elaine = Image.open(path+'elaine.tiff')
	(mags,rads,_) = ftransform(elaine)
	plt.imshow(rads, cmap='Greys_r')
	plt.show()
	rads *= 1.75
	com = p2c(mags,rads)
	aux = np.abs(np.fft.ifft2(np.fft.ifftshift(com)))
	plt.imshow(aux, cmap='Greys_r')
	plt.show()


# Questão 6
def q6(path):
	print('Questão 6')
	lena = Image.open(path+'lena.tif')
	(mags,rads,_) = ftransform(lena)
	rads *= -1
	com = p2c(mags,rads)
	aux = np.abs(np.fft.ifft2(np.fft.ifftshift(com)))
	plt.imshow(aux, cmap='Greys_r')
	plt.show()


# filtragem homomórfica
def fhomom(im, gl, gh, c, d):
	image = np.fft.fft2(np.log(im))
	P = 2*im.size[0]
	Q = 2*im.size[1]
	H = np.zeros((image.shape), dtype=complex)
	for u in range(image.shape[0]):
		for v in range(image.shape[1]):
			H[u,v] = image[u,v]*((gh-gl)*(1-np.exp(-c*(np.square(u-P/2)+np.square(v-Q/2))/np.square(d)))+gl)
			# H[u,v] = 1/np.power(1+np.sqrt(np.square(u-P/2)+np.square(v-Q/2))/d,4)
	return np.exp(np.abs(np.fft.ifft2(H)))


# Questão 7
def q7(path):
	print('Questão 7')
	# maril = Image.open(path+'mar-il.gif')
	# plt.plot(maril.histogram())
	# plt.show()
	# aux = fhomom(maril,0.25,2,1,80)
	# print(aux)
	# maril.putdata(aux)
	# plt.imshow(aux, cmap='Greys_r')
	# plt.show()
	# plt.plot(maril.histogram())
	# plt.show()

	image = Image.open(path+'image.png').convert('L')
	plt.plot(image.histogram())
	plt.show()
	aux = fhomom(image,0.25,2,1,80)
	image.putdata(aux)
	plt.imshow(aux, cmap='Greys_r')
	plt.show()
	plt.plot(image.histogram())
	plt.show()


def main(path='/Users/mthome/Dropbox/UFES/Processamento Digital de Imagens/lista1/'):
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
	q7(path)

if(__name__ == '__main__'):
	main()
