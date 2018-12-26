#!/usr/bin/env python3

import numpy as np
from PIL import Image
import math
import sys
import os

def main(path='/Users/mthome/Dropbox/UFES/Processamento Digital de Imagens/trab/'):
	try:
		os.mkdir(path+'output/')
	except FileExistsError as e:
		pass
	except:
		print('error while creating output dir')

if(__name__ == '__main__'):
	main()
