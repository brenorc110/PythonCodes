import numpy as np
import cv2

global img
class Controle:
    #Função para facilitar a escrita nas imagem
    def main(self):
        imagemCarregada = cv2.imread('dados.jpg') #Carregamento da imagem
        #Se necessário o redimensioamento da imagem pode vir aqui.

        #Passo 1: Conversão para escala de cinza: 
        img = cv2.cvtColor(imagemCarregada, cv2.COLOR_BGR2GRAY)

        #Passo 2: Blur/Suavização da imagem
        suave = cv2.GaussianBlur(img, (7, 7), 0)
        #Passo 3: Binarização resultando em pixels brancos e pretos
        #1- Imagem em tons de cinza
        #2- Suavizacao com Blur
        #3- Binarizacao com Metodo Otsu
        #4- Detector de bordas Canny
    
        (T, bin) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY)
        (T, binI) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY_INV)
        resultado = np.vstack([
        np.hstack([suave, bin]),
        np.hstack([binI, cv2.bitwise_and(img, img, mask = binI)])
        ])
        
        cv2.imshow("Binarização da imagem", resultado)
        bin = cv2.bitwise_not(bin)
        #Passo 4: Detecção de bordas com Canny
        bordas = cv2.Canny(bin, 70, 150)
        #Passo 5: Identificação e contagem dos contornos da imagem
        #cv2.RETR_EXTERNAL = conta apenas os contornos externos
        (objetos, lx) = cv2.findContours(bordas.copy(),
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for geo in objetos:
            cv2.drawContours(img, [geo], 0, (255, 0, 0), 3)
        
        #A variável lx tem o vetor das bordas, um poligono  que representa o triangulo.
        #VEr função  drawcontorn.
        temp = np.vstack([
            np.hstack([img, suave]),
            np.hstack([bin, bordas])
            ])
        cv2.imshow("Quantidade de objetos: "+str(len(objetos)), img)
        #imgC2 = img.copy()
        #cv2.imshow("Imagem Original", imagemCarregada)
        #cv2.drawContours(imgC2, objetos, -1, (255, 0, 0), 2)
        #cv2.imshow(imgC2, str(len(objetos))+" objetos encontrados!")
        #cv2.imshow("Resultado", imgC2)
        cv2.waitKey(0)
        #cv2.imshow("Canal verde", cv2.merge([zeros, canal_v, zeros]))

controle = Controle()
controle.main()
