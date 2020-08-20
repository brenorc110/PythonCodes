import numpy as np
from scipy.stats import mode
import cv2
import os
import csv
import time

#Altere para o cascade treinado
cascade_treinado = cv2.CascadeClassifier('C:/Users/Breno/Documents/VS Code Projects/Python/haarcascades/cascade.xml')
cascade_treinado03 = cv2.CascadeClassifier('C:/Users/Breno/Documents/VS Code Projects/Python/haarcascades/cascade03.xml')

class Controle:   
    def main(self):
        #carrega um classificador de um arquivo
        aux = 0
        aux1= 1
        maior = []
        #Para cada arquivo na lista faça:

        #carrega um vídeo
        print("[INFO] Inicializando webcam... ")
        cap = cv2.VideoCapture(0)
        cap.set(3, 800) # Set largura
        cap.set(4, 600) # Set altura

        while(not cv2.waitKey(1) & 0xFF == ord('q')):
            aux = 0
            aux1= 1
            #carrega o frame de vídeo
            frameExiste, frame = cap.read()
            
            #chegou ao último frame ou houve erro? então sair!
            if(frameExiste == False):
                print("Algo de errado não está certo com a câmera!!!")
                cap.release()
                return
            
            #somente funciona com tons de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #Faz as classificações
            cascade = cascade_treinado.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
            cascade03 = cascade_treinado03.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

            #Organiza numa as classificações numa lista para loop
            classifiers = [cascade, cascade03]

            for (classifier) in classifiers:
                test = format(len(classifier))
                aux1 = int(test)
                print("Dados:",test)
                
                if aux1 > aux:
                       aux = aux1
                       print ("Melhor Resultado:",aux)
                       maior = (classifier)
                
            # Coloca os quadrados nos objetos
            for (x, y, w, h) in maior:
                #desenhe um retângulo (imagem, posição inicial, final, cor, espessura)
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255), 2)
                #roi_gray = gray[y:y+h, x:x+w]
                #roi_color = frame[y:y+h, x:x+w]
                #time.sleep(0.3)

            #visualizar o detectado: 
            cv2.imshow("deteccao", frame)            
        #Desliga webcam
        cap.release()
        #Destroi janela
        cv2.destroyAllWindows()
        
controle = Controle()
controle.main()
