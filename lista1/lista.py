#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image
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
	;


# Questão 2
def q2(path):
	print('Questão 2')
	template = Imagem.open(path+'template.jpg')



def main(path='/Users/mthome/Dropbox/UFES/Processamento Digital de Imagens/lista1/'):
	try:
		os.mkdir(path+'output/')
	except FileExistsError as e:
		pass
	except:
		print('error while creating output dir')

	q1(path)

if(__name__ == '__main__'):
	main()
