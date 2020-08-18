#import cv2
#import numpy as np
#imagemcarregada = cv2.imread("corshow.png",1) # Transferimos o conteúdo da img para a variável 'imagemcarrregada'.
#0 carrega em PeB, 1 em colorido. 

#cv2.imshow("imagem carregada: ", imagemcarregada) # nome da janela para exibição da imagem.
# função modifica pixel (em ordem bgr) pixel 0,0 de cada cor: 
#imagemcarregada.itemset((0, 0, 2), 255)
#imagemcarregada.itemset((0, 0, 1), 0)
#imagemcarregada.itemset((0, 0, 1), 0)

#cv2.imshow("imagem carregada 2: ", imagemcarregada)
# inserir redimensionamento de imagens aqui.
#cv2.waitKey(0) # espera clique do usuário, 0 espera infinitamente. 
#cv2.destroyAllWindows()#Destroi as janelas.

#tamanho da imagem: 
#print('Largura em pixels: ', end='')
#print(imagem.shape[1]) 
#print('Altura em pixels: ', end='')  
#print(imagem.shape[0]) 
#print('Qtde de canais: ', end='')  
#print(imagem.shape[2]) 

# valores das cores dos canais b, g e r (ler pixel).
#print (imagemcarregada.item(0, 0, 2), imagemcarregada.item(0, 0, 2), imagemcarregada.item(0, 0, 2))

#cv2.imwrite("imagemsaida.jpg", imagemcarregada) #salvando imagem modificada.

#recorte = imagemcarregada[180:250, 250:315]#yi:yf, xi:xf
#cv2.imwrite("imagemrecortada", recorte)#salva nova imagem de acordo com as coordenadas estabelecidaas anteriormente

# redimensionando imagem:
#imagemredimensionada = cv2.resize(imagemcarregada,(600, 600)) # exibe somento o pedaço de 600x600 px
#imagemredimensionada = cv2.resize(imagemcarregada, (int(imagemcarregada.shape[1]*2), int(imagemcarregada.shape[0]*2))) # dobra img de tamanho.

# Desenha um retangulo nos elementos verdes detectados: 
#for (x, y, w, h) in verde:
#    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 4)

import cv2 
import numpy as np  

def show_image():
    # Webcamera no 0 is used to capture the frames 
    #cap = cv2.VideoCapture(0)  
    imagemcarregada = cv2.imread("jorppoes.jpg",1)
    # This drives the program into an infinite loop. 
    # Captures the live stream frame-by-frame 
    #_, frame = cap.read()  
    # Converts images from BGR to HSV 
    hsv = cv2.cvtColor(imagemcarregada, cv2.COLOR_BGR2HSV) 
    lower_blue = np.array([110,50,50]) 
    upper_blue = np.array([130,255,255]) 
      
    # Here we are defining range of bluecolor in HSV 
    # This creates a mask of blue coloured  
    # objects found in the frame. 
    mask = cv2.inRange(hsv, lower_blue, upper_blue) 
      
    # The bitwise and of the frame and mask is done so  
    # that only the blue coloured objects are highlighted  
    # and stored in res 
    res = cv2.bitwise_and(imagemcarregada,imagemcarregada, mask= mask) 
    #cv2.imshow('frame',frame) 
    cv2.imshow('Mascara (Azul)',mask) 
    cv2.imshow('imagemCarregada',imagemcarregada)
    
    #gray = cv2.cvtColor(imagemcarregada, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Escala de cinza',gray)

    (canalAzul, canalVerde, canalVermelho) = cv2.split(imagemcarregada)
    zeros = np.zeros(imagemcarregada.shape[:2], dtype = "uint8")

    cv2.imshow("Azul", cv2.merge([canalAzul, zeros, zeros]))
    #cv2.imshow("Verde", cv2.merge([zeros, canalVerde, zeros]))
    #cv2.imshow("Vermelho", cv2.merge([zeros, zeros, canalVermelho]))
    
    # This displays the frame, mask  
    # and res which we created in 3 separate windows. 
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        pass

cv2.waitKey(0) # espera clique do usuário, 0 espera infinitamente. 
cv2.destroyAllWindows()#Destroi as janelas.

def main():
    show_image()
    return 0
 
if __name__ == '__main__':
    main()
