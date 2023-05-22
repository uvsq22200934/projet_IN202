def decompresse(liste_blocs_Y, liste_blocs_Cb, liste_blocs_Cr):
    hauteur = len(liste_blocs_Y) * 8
    largeur = len(liste_blocs_Y[0]) * 8

    # Créer une matrice pour chaque canal
    mat_Y = np.zeros((hauteur, largeur), dtype=np.float32)
    mat_Cb = np.zeros((hauteur, largeur), dtype=np.float32)
    mat_Cr = np.zeros((hauteur, largeur), dtype=np.float32)

    # Reconstruire chaque bloc dans les matrices de canal
    for i in range(len(liste_blocs_Y)):
        for j in range(len(liste_blocs_Y[0])):
            mat_Y[i*8:(i+1)*8, j*8:(j+1)*8] = liste_blocs_Y[i][j]
            mat_Cb[i*8:(i+1)*8, j*8:(j+1)*8] = liste_blocs_Cb[i][j]
            mat_Cr[i*8:(i+1)*8, j*8:(j+1)*8] = liste_blocs_Cr[i][j]

    # Convertir les matrices de canal en valeurs entières dans la plage RGB
    mat_Y = np.clip(np.round(mat_Y + 128), 0, 255).astype(np.uint8)
    mat_Cb = np.clip(np.round(mat_Cb + 128), 0, 255).astype(np.uint8)
    mat_Cr = np.clip(np.round(mat_Cr + 128), 0, 255).astype(np.uint8)

    # Convertir les matrices Y, Cb, Cr en une matrice RGB
    mat_RGB = np.empty((hauteur, largeur, 3), dtype=np.uint8)
    mat_RGB[:, :, 0] = mat_Y
    mat_RGB[:, :, 1] = mat_Cb
    mat_RGB[:, :, 2] = mat_Cr

    return mat_RGB





def decompresse(blocsY, blocsCb, blocsCr):

#### enleve blocs et remets en matrice
    l = len(blocsY) * 8
    c = len(blocsY[0]) * 8

    matY = np.zeros((l,c), dtype=np.float32)
    matCb = np.zeros((l,c), dtype=np.float32)
    matCr = np.zeros((l,c), dtype=np.float32)

    for i in range(len(blocsY)):
        for j in range(len(blocsY[O])):
            matY[i*8:(i+1)*8, j*8:(j+1)*8] = blocsY[i][j]
            matCb[i*8:(i+1)*8, j*8:(j+1)*8] = blocsCb[i][j]
            matCr[i*8:(i+1)*8, j*8:(j+1)*8] = blocsCr[i][j]

    matYCbCr = [matY, matCb, matCr]

    mat_doublee

    im = elimine_padding(matYCbCr)

    image = RGB(im)

            
            