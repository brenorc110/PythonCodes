import os
import numpy as np
import cv2

class Controle:
    
    def menu(self):
        txt = """
        ESCOLHA A COR A SER DETECTADA:
            1-Vermelho
            2-Verde
            3-Azul
            4-Sair
        Digite a sua opção
        """
        opcao = int(input(txt))
        return opcao

#FUNÇÃO MAIN PEDE O NOME DA IMAGEM E CONVERTE PARA HSV

    def main(self):
        while True:  #Input e procura do nome da imagem (precisa estar na mesa pasta que o .py)
            #img = input("Digite o nome da imagem com a extensao EX: img.ext : ")
            #file = os.path.realpath(img)
            
            global imagem,imagem1 #variável GLOBAL
            imagem = cv2.imread('A.jpg')#Lê A imagem
            imagem1 = cv2.imread('A.jpg')
            
            #Converte de BGR para HSV
            try:
                global hsv #variável GLOBAL
                hsv = cv2.cvtColor(imagem,cv2.COLOR_BGR2HSV) #Conversão BGR para HSV
                break
            except:
                print("Imagem nao encontrada")

            
#////////////////MENU/////////////////////
    #Menu de escolha de cor pra achar           
        opcoes = {1:self.opcao1, 2:self.opcao2, 3:self.opcao3}
        while True:
            opcao = self.menu()
            if opcao in opcoes:
                opcoes[opcao]()
                break
            else:
                if opcao == 4:
                    break
                else:
                    print("Opção inválida")

#////////////////DETECÇÃO DE CORES E GERAÇÃO DO RESULTADO/////////////////////
    
    def opcao1(self):  #Detecta VERMELHO
        cor=1
        ulCor.maskRes(ulCor.rColor(0,10,10),ulCor1.rColor(200,255,255))

    def opcao2(self): #Detecta VERDE
        ulCor.maskRes(ulCor.rColor(20,20,20),ulCor1.rColor(80,255,255))

    def opcao3(self): #Detecta AZUL
        ulCor.maskRes(ulCor.rColor(75,100,100),ulCor1.rColor(150,255,255))


#Classe que trata o processamento da máscara e o resultado
class ulColor(object):

    #Range de detecão de Cor
    def rColor(self, h, s, v):
        self.rangeUL=np.array([h,s,v],np.uint8)
        return self.rangeUL

    #Gera a mascara, pede 2 instancias da classe para usar método acima
    def maskRes(self, ulCor, ulCor1):    
        
        #GERA MASCARA 
        mask = cv2.inRange(hsv, ulCor, ulCor1)
        mask = cv2.GaussianBlur(mask, (1, 1), 0)
        res = cv2.bitwise_and(imagem,imagem, mask=mask)
        resO = cv2.bitwise_and(imagem,imagem, mask=mask)
        
        #GERA MASCARA INVERTIDA
        mask_inv = cv2.bitwise_not(mask)
        res1 = cv2.bitwise_and(imagem,imagem, mask=mask_inv)
        
        zeros = np.zeros(res.shape[:2], dtype = "uint8")
        test = cv2.bitwise_not(mask)

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        suave = cv2.GaussianBlur(gray, (7, 7), 0)

        bordas = cv2.Canny(mask_inv, 5, 150)
        (objetos, lx) = cv2.findContours(bordas.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #Desenha contorno amarelo na mascara invertida (20,202,255)
        t2 = cv2.drawContours(imagem1, objetos, -1, (20,202, 255), 2)
        #cv2.imshow("Quantidade de objetos: "+str(len(objetos)), imagem)
        
        #Divisão da area selecionada em canais RGB
        (canalAzul, canalVerde, canalVermelho) = cv2.split(res)

        res1 = cv2.bitwise_and(imagem,imagem, mask=mask_inv)
        test = cv2.bitwise_not(mask)

        v1 = -1
        v2 = -1
        v3 = -1
        
        #Inserir valor dos multiplicadores
        while (v1 <0 or v1 >255): 
            v1 = int(input("Insira Multiplicador Canal [R] Entre 0 e 255: "))
        while (v2 <0 or v2 >255): 
            v2 = int(input("Insira Multiplicador Canal [G] Entre 0 e 255: "))
        while (v3 <0 or v3 >255): 
            v3 = int(input("Insira Multiplicador Canal [B] Entre 0 e 255: "))
        
        #JUNTA O RGB E MUDA A COR - VARIA DE 0 A 255
        res = cv2.merge([canalAzul*v3, canalVerde*v2, canalVermelho*v1])

        #Com a imagem do dado olhe esse parâmetro:
        #res = cv2.merge([canalAzul*0, canalVerde*255, canalVermelho*0])
        
        #Junta A mascara Invertida e a area REcolorida
        final = cv2.add(res,res1)
        #final = cv2.add(res,imagem)

        temp = np.vstack([
            np.hstack([t2, resO]),
            np.hstack([res, final]),
            ])

        #temp1 = np.vstack([np.hstack([mask, mask_inv]),])
        #cv2.imshow("Imagem ORIGINAL", imagem)
        cv2.imshow("Quantidade de objetos: "+str(len(objetos)), temp)
        #cv2.imshow("MN", temp1)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows
        

#Instâncias
ulCor = ulColor()
ulCor1 = ulColor()
controle = Controle()
controle.main()
