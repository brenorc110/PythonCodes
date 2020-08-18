import numpy as np
from scipy.stats import mode
import cv2
import os
import csv
import time
import pathlib

class Controle:
    #Pega o caminho das pastas automaticamente
    path = os.path.dirname(os.path.abspath(__file__)) +'\\Entrada'
    path1 = os.path.dirname(os.path.abspath(__file__)) +'\\positive'
    path2 = os.path.dirname(os.path.abspath(__file__)) +'\\negative'
    path3 = os.path.dirname(os.path.abspath(__file__)) +'\\Saida'
    maior = []
    
    os.chdir(path)
    def menu(self):
        txt = """
        Opções:
            1-Reconhecimento por camera
            2-Reconhecimento por foto
            3-Reconhecimento com o casacade treinado
            4- Sair
        Digite a sua opção:
        """
        opcao = int(input(txt))
        return opcao

    def main(self):
        opcoes = {1:self.opcao1, 2:self.opcao2, 3:self.opcao3}
        while True:
            opcao = self.menu()
            if opcao in opcoes:
                opcoes[opcao]()
            else:
                if opcao == 4:
                  break
                else:
                  print("Opção inválida")
                  
    def opcao1(self):
        #carrega um classificador de um arquivo
        aux = 0
        aux1= 1
        maior =[]
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
            
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50,50))
            faces2 = face_alt_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50,50))
            faces3 = face_alt2_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50,50))
            faces4 = face_alt_tree_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50,50))

            #Organiza numa as classificações numa lista para loop
            classifiers = [faces, faces2, faces3, faces4]

            for (classifier) in classifiers:
                test = format(len(classifier))
                aux1 = int(test)
                print("Faces:",test)
                if aux1 > aux:
                   aux = aux1
                   print ("Melhor Resultado:",aux)
                   maior = (classifier)
				     
            #para cada face detectada
            for (x, y, w, h) in maior:
                #desenhe um retângulo (imagem, posição inicial, final, cor, espessura)
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255), 2)
                #time.sleep(0.3)
            #visualizar o detectado: 
            cv2.imshow("deteccao", frame)
            
        #Desliga webcam
        cap.release()
        #Destroi janela
        cv2.destroyAllWindows()

    def opcao2(self):
        #=== caminhos ===
        #path ../Entrada
        #path1 ../training/positive
        #path2 ../training/negative
        #path3 ../Saida
        maior = []
        file_list = []
        file_list = busca_diretorio(controle.path,".jpg")

        if  len(file_list) != 0:
            for file in file_list:
                print("[INFO] Buscando as imagens na pasta padrão... ")
                aux = 0
                aux1= 1
                #Para cada arquivo na lista faça:
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
                classifiers = [faces, faces2, faces3, faces4,faces5, faces6, faces7, faces8]

                for (classifier) in classifiers:
                    test = format(len(classifier))
                    aux1 = int(test)
                    print("Faces:",test)
                    if aux1 > aux:
                       aux = aux1
                       print ("Melhor Resultado:",aux)
                       maior = (classifier)
                       
                # Coloca os quadrados nas faces
                for (x,y,w,h) in maior:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]
                              
                #Exibe as imagens com retangulos, salva imagem com retangulos se ela não existe, so mostra se ela existe

                os.chdir(controle.path) #Diretorio ativo no momento
                existe = pathlib.Path(file)
                                      
                if existe.exists() is True:
                    cv2.imshow('img',img)
                    print("Para a imagem "+file+", foram encontradas {0} faces!".format(len(maior)))
                else:
                    cv2.imshow('img',img)
                    print("Para a imagem "+file+", foram encontradas {0} faces!".format(len(maior)))
                    cv2.imwrite(file,img)
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                os.chdir(controle.path)
               
            cv2.destroyAllWindows()
            #controle.saida.close()
        else:
            print("[INFO] Pasta Vazia ")

    def opcao3(self):
        file_list = []
        file_list = busca_diretorio(controle.path,".jpg")
        
        if  len(file_list) != 0:
            for file in file_list:
                print("[INFO] Buscando as imagens na pasta padrão... ")
                aux = 0
                aux1= 1
                #Para cada arquivo na lista faça:
                #Lê a imagem e converte para escala de cinza
                img = cv2.imread(file)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            
                #Faz as classificações
                cascade = cascade_treinado.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
                
                #Organiza numa as classificações numa lista para loop
                classifiers = [cascade]

                for (classifier) in classifiers:
                    test = format(len(classifier))
                    aux1 = int(test)
                    print("Objetos:",test)
                    
                # Coloca os quadrados nos objetos
                for (x,y,w,h) in classifier:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]
                              
                #Exibe as imagens com retangulos, salva imagem com retangulos se ela não existe, so mostra se ela existe
                os.chdir(controle.path) #Diretorio ativo no momento
                existe = pathlib.Path(file)
                                      
                if existe.exists():
                    cv2.imshow('img',img)
                    print("Para a imagem "+file+", foram encontrados {0} dados.".format(len(classifier)))
                else:
                    cv2.imshow('img',img)
                    print("Para a imagem "+file+", foram encontrados {0} dados.".format(len(classifier)))
                    cv2.imwrite(file,img)
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                os.chdir(controle.path)
               
            cv2.destroyAllWindows()
        else:
            print("[INFO] Pasta Vazia ")

#=================================================
#======== VARIÁVEIS E MÉTODOS GLOBAIS ============
#=================================================
 
#Estabelece os classificadores de face

#altere para o cascade treinado
cascade_treinado = cv2.CascadeClassifier('../haarcascades/cascade.xml')
cascade03 = cv2.CascadeClassifier('../haarcascades/cascade03.xml')

face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
face_alt_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')
face_alt2_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt2.xml')
face_alt_tree_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt_tree.xml')

#funcao de busca de arquivos no diretório
def busca_diretorio(diretorio,extensao):
    file_list = []
    for file in os.listdir(diretorio):
        if file.endswith(extensao):
            file_list.append(file)
            #print(file)
    return file_list

controle = Controle()
controle.main()
