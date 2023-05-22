import PIL as pil
from PIL import Image, ImageTk
import numpy as np
import scipy as sp
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
import random
from math import log10, sqrt

def load(filename):
    toLoad= Image.open(filename)
    return np.asarray(toLoad)


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def dct2(a):
    return sp.fft.dct( sp.fft.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return sp.fft.idct( sp.fft.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

nomImg=""


########################
# BLOC
########################

def YCbCr(mat):
    matYCbCr=np.empty((mat.shape[0],mat.shape[1],3),dtype=np.float64)

    for i in range(matYCbCr.shape[0]):
        for j in range(matYCbCr.shape[1]):
            matYCbCr[i,j,0] = 0.299 * mat[i,j,0] + 0.587 * mat[i,j,1] + 0.114 * mat[i,j,2]
            
            matYCbCr[i,j,1] = - 0.1687 * mat[i,j,0] - 0.3313 * mat[i,j,1] + 0.5 * mat[i,j,2] + 128

            matYCbCr[i,j,2] = 0.5 * mat[i,j,0] - 0.4187 * mat[i,j,1] - 0.0813 * mat[i,j,2] + 128
            
    return matYCbCr

########################
# BLOC
########################


def RGB(mat):
    matRGB=np.empty([mat.shape[0],mat.shape[1],3],dtype=np.uint8)

    for i in range(matRGB.shape[0]):
        for j in range(matRGB.shape[1]): 
            
            y = mat[i,j,0]
            cb = mat[i,j,1]
            cr = mat[i,j,2]
            
            r = y + 1.402 * (cr - 128) 
            g = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
            b = y + 1.772 * (cb - 128)
            
            matRGB[i,j,0] = int(r)
            
            matRGB[i,j,1] = int(g)
            
            matRGB[i,j,2] = int(b)
             
    return matRGB


def padding(mat):
    l = mat.shape[0]
    c = mat.shape[1]
    while l % 8 != 0:
        l += 1
    while c % 8 !=0:
        c += 1
    mat_pad= np.zeros((l, c, 3), dtype=np.uint8)
    mat_pad[:mat.shape[0],:mat.shape[1]]=mat
    return mat_pad
    
def elimine_padding(mat_padding):
    l_pad, c_pad, d = mat.shape
    l = l_pad
    c = c_pad
    while l % 8 != 0:
        l -= 1
    while c % 8 != 0:
        c -=1
    mat_sans_padding=np.empty((l,c,3),dtype=np.uint8)
    for i in range(l):
        for j in range(c):
            mat_sans_padding[i,j] = mat[i,j]
    return mat_sans_padding


########################
# BLOC
########################


def petite_mat(mat):
    mat_reduite=np.empty((len(mat), len(mat[0])//2,3),dtype=np.uint8)
    for i in range(0,len(mat_reduite)):
        for j in range(0, len(mat_reduite[0]), 2):
            mat_reduite[i,j,1] = (mat[i,2*j,1] + mat[i,2*j+1,1]) // 2
            mat_reduite[i,j,2] = (mat[i,2*j,2] + mat[i,2*j+1,2]) // 2
    return mat_reduite


########################
# BLOC
########################


def matrice_doublee(mat):
    mat_doublee =np.empty([(mat.shape[0]), mat.shape[1]*2, 3], dtype=np.uint8)
    for i in range(mat.shape[0]):
        for j in range(0,mat.shape[1],2):
            mat_doublee[i,j] = mat[i,j//2]
            mat_doublee[i,j+1] = mat[i,j//2]
    return mat_doublee


########################
# BLOC
########################


def blocs(mat):
    liste_blocs =[]
    for element in range(3):
        for i in range(0, mat.shape[0], 8):
            for j in range (0, mat.shape[1], 8) :
                bloc = mat[i:i+8, j:j+8, element]
                liste_blocs.append(bloc)
                
    return liste_blocs

########################
# BLOC
########################

def transformee(blocs):
    blocs_transforme = []
    for i in range(len(blocs)):
        u = dct2(blocs[i])
        blocs_transforme.append(u)
    return blocs_transforme

def detransformation(blocs):
    blocs_detransformes = []
    for i in range(len(blocs)):
        blocs_detransformes.append(idct2(blocs[i]))
    return blocs_detransformes 

########################
# BLOC
########################

def filtrage1(blocs,seuil):
    for i in range(len(blocs)):
        if blocs[i] < seuil :
            blocs[i] = 0
            
########################
# BLOC
########################
           

def blocs_compresses(image, mode):
    image = YCbCr(image)
    if mode == 2:
        image = petite_mat(image)
    image = blocs(padding(image))
    image = transformee(image)
    #im = YCbCr(padding(image.all()))
    if mode > 0:
        image = filtrage1(image, 100)
    return image

########################
# BLOC
########################

def rle(texte):
    texte_rle = []
    i = 0
    k = 0

    while i < len(texte):
        if texte[i] != '0':
            texte_rle.append(texte[i])
        elif texte[i] == '0' :
            i += 1
            while texte[i] == '0':
                k += 1
            texte_rle.append('#',k)
        i += 1
    return texte_rle

def write_file(path: str, mode: int, rle: bool):

    mat = load(path)
    l = blocs_compresses(mat, mode)
    print(len(l[0]), len(l[0][0]))

    with open('f','w') as f:
        f.write('SJPG\n')
        f.write(f"{mat.shape[0]} {str(mat.shape[1])}\n")

        f.write(f'mode {mode}\n')
        if not rle:
            f.write('NO')
        f.write('RLE\n')

        y = []
        cb = []
        cr = []

        for i in range(len(l)):
            for y in range(len(l[i])):
                for x in range(len(l[i][0])):
                    f.write(f"{int(l[i][y][x])} ")
            f.write("\n")

import pickle 
def lectSJPG (fichier):
    liste1 = []
    f = open(fichier , 'r')
    type = f.readline()
    h = int((f.read(4)).strip())
    l = int(f.read(3))
    print(h,l)
    mode = f.readline()
    RLE = f.readline()
    liste1 = np.empty((h,l,3),dtype=np.uint8)
    for i in range(h): 
        for j in range(l):
            for k in range(64):
                pixely= f.readline().strip().split()
                pixelcb = f.readline().strip().split()
                pixelcr = f.readline().strip().split()
                liste1[i,j]=[int(pixely[k]),int(pixelcb[k]),int(pixelcr[k])]      
                #liste1.append(blocs[i][j])         
    f.close()

lectSJPG ('no_compression')

#write_file("test1.png", 2, False)

3078/3=1026 











def decompression(blocsY, blocsCb, blocsCr):

## - RLE ##

    blocs = [blocsY, blocsCb, blocsCr]

    l =[]
    for i in range(len(blocs)):
        for j in range(blocs[i]):
            if blocs[i,j] != '#':
                l.append(blocs[i,j])
            elif blocs[i,j] == '#':
                z =[]
                n = 0
                while n != ' ':
                    z.append(blocs[i,j+1])
                    n += 1
                l.append(str(z))   


    d = detransformation(l)

## remettre blocs en matrice

    matYCbCr = np.empty((len(d)//2, len(d)//2 3), dtype = np.uint8)
    for n in range(0,len(d),8):
    

def 





import numpy as np
import scipy.fft as fft

def decompress_image(Y_blocks, Cb_blocks, Cr_blocks):
    # calcul de dimentions
    blocks_longeur, blocks_largeur = len(Y_blocks), len(Y_blocks[0])
    image_height = blocks_longeur * 8
    image_width = blocks_largeur * 8

    # Initialisation des mat
    Y_matrix = np.zeros((image_height, image_width))
    Cb_matrix = np.zeros((image_height, image_width))
    Cr_matrix = np.zeros((image_height, image_width))

    
    for i in range(blocks_longeur):
        for j in range(blocks_largeur):
            
            Y_block = idct2(Y_blocks[i][j])
            Y_matrix[i*8:(i+1)*8, j*8:(j+1)*8] = Y_block

            
            Cb_block = idct2(Cb_blocks[i][j])
            Cr_block = idct2(Cr_blocks[i][j])
            Cb_matrix[i*8:(i+1)*8, j*8:(j+1)*8] = Cb_block
            Cr_matrix[i*8:(i+1)*8, j*8:(j+1)*8] = Cr_block

    # Convertion
    R_matrix = Y_matrix + 1.402 * (Cr_matrix - 128)
    G_matrix = Y_matrix - 0.344136 * (Cb_matrix - 128) - 0.714136 * (Cr_matrix - 128)
    B_matrix = Y_matrix + 1.772 * (Cb_matrix - 128)

    
    R_matrix = np.clip(R_matrix, 0, 255).astype(np.uint8)
    G_matrix = np.clip(G_matrix, 0, 255).astype(np.uint8)
    B_matrix = np.clip(B_matrix, 0, 255).astype(np.uint8)

    
    return np.dstack((R_matrix, G_matrix, B_matrix))





from PIL import Image

def read_jpeg_file(file_path):
    image = Image.open(file_path)

    
    image_ycbcr = image.convert("YCbCr")

    
    image_width, image_height = image_ycbcr.size

    
    num_blocks_width = image_width // 8
    num_blocks_height = image_height // 8

    Y_blocks = [[None] * num_blocks_width for _ in range(num_blocks_height)]
    Cb_blocks = [[None] * num_blocks_width for _ in range(num_blocks_height)]
    Cr_blocks = [[None] * num_blocks_width for _ in range(num_blocks_height)]
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            
            Y_block = image_ycbcr.crop((j*8, i*8, (j+1)*8, (i+1)*8)).load()[0, 0]
            Cb_block = image_ycbcr.crop((j*8//2, i*8//2, (j+1)*8//2, (i+1)*8//2)).load()[0, 0]
            Cr_block = image_ycbcr.crop((j*8//2, i*8//2, (j+1)*8//2, (i+1)*8//2)).load()[0, 1]

            # Stocker les blocs dans les listes de blocs
            Y_blocks[i][j] = Y_block
            Cb_blocks[i][j] = Cb_block
            Cr_blocks[i][j] = Cr_block

    
    return Y_blocks, Cb_blocks, Cr_blocks


# Exemple de valeurs à remplacer par vos propres informations
file_type = "SJPG"
image_height = 200
image_width = 300
compression_mode = "mode 1"
compression_type = "RLE"


with open("f", "w") as f:
    f.write(file_type + "\n")
    f.write(str(image_height) + " " + str(image_width) + "\n")
    f.write(compression_mode + "\n")
    f.write(compression_type + "\n")
mat=load("test.png")
compress_image(mat,2)




with open("f", "w") as f:
    f.write('SJPG\n')
    f.write(str(mat.shape[0]))
    f.write(' ')
    f.write(str(mat.shape[1]))
    f.write('\nmode 2')
    f.write('\nNORLE\n')
    
    f.close()

    