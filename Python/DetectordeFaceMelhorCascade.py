import numpy as np
from scipy.stats import mode
import cv2
import os
import csv
import time
path='C:/Users/Breno/Documents/VS Code Projects/Python'
os.chdir(path)

#Salva o arquivo com titulo atualizado por data e hora
titulo = time.strftime("%Y-%m-%d") + '_' + time.strftime("%H-%M-%S")
saida = open('face_recon_'+titulo+'.csv', 'w')
export = csv.writer(saida, quoting=csv.QUOTE_NONNUMERIC)

maior=0
file_list = []

for file in os.listdir(path):
    if file.endswith(".jpg"):
        file_list.append(file)
        #print(file_list)
for file in file_list:
    aux =0
    aux1 =1
    #Estabelece os classificadores de face
    face_cascade = cv2.CascadeClassifier('C:/Users/bcunha/OneDrive/Python Codes/reconheceObjetosIA/haarcascades/haarcascade_frontalface_default.xml')
    face_alt_cascade = cv2.CascadeClassifier('C:/Users/bcunha/OneDrive/Python Codes/reconheceObjetosIA/haarcascades/haarcascade_frontalface_alt.xml')
    face_alt2_cascade = cv2.CascadeClassifier('C:/Users/bcunha/OneDrive/Python Codes/reconheceObjetosIA/haarcascades/haarcascade_frontalface_alt2.xml')
    face_alt_tree_cascade = cv2.CascadeClassifier('C:/Users/bcunha/OneDrive/Python Codes/reconheceObjetosIA/haarcascades/haarcascade_frontalface_alt_tree.xml')

    #Lê a imagem e converte para escala de cinza
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Faz as classificações
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    faces2 = face_alt_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    faces3 = face_alt2_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    faces4 = face_alt_tree_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

    faces5 = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20,20))
    faces6 = face_alt_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20,20))
    faces7 = face_alt2_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20,20))
    faces8 = face_alt_tree_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20,20))

    #Organiza numa as classificações numa lista para loop
    classifiers = [faces, faces2, faces3, faces4, faces5, faces6, faces7, faces8]

    #Testa qual Classificador teve o melhor resultado
    for classifier in classifiers:
        test = format(len(classifier))
        aux1 = int(test)
        print("Faces:",test)
        if aux1 > aux:
            aux = aux1
            print ("Melhor Resultado:", aux)
            maior = (classifier)
            
    #for (classifier) in classifiers:
        #Coloca os quadrados nas faces
        for (x,y,w,h) in maior:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

    print("Para a imagem "+ file +", foram encontradas {0} faces!".format(len(maior)))
    #Exibe as imagens com quadrados.
    cv2.imshow('img',img)
    cv2.waitKey(0)

    # Exibe a média, variância e moda de cada classificador
    encontrados = []

    for (classifier) in classifiers:
        x = format(len(classifier))
        encontrados.append(x)

    encontrados = np.asarray(encontrados, dtype = np.float16)
    media = np.mean(encontrados)
    variancia = np.var(encontrados)
    moda = float(mode(encontrados)[0])

    if file == file_list[0]:
        export.writerow(["imagem","media","variancia","moda"])
        export.writerow([file, media, variancia, moda])
    else:
        export.writerow([file, media, variancia, moda])
#Destroi as janelas.
cv2.destroyAllWindows()
saida.close()
