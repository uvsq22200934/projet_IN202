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

write_file("test1.png", 0, False)
