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
            4-sair
        Digite a sua opção
        """
        opcao = int(input(txt))
        return opcao

#FUNÇÃO MAIN PEDE O NOME DA IMAGEM E CONVERTE PARA HSV

    def main(self):
        while True:  #Input e procura do nome da imagem (precisa estar na mesa pasta que o .py)
            img = input("Digite o nome da imagem com a extensao EX: img.ext : ")
            file = os.path.realpath(img)
            
            global imagem #variável GLOBAL
            imagem = cv2.imread(img)#Lê A imagem
            
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
        
        #GERA MASCARA INVERTIDA
        mask_inv = cv2.bitwise_not(mask)
        res1 = cv2.bitwise_and(imagem,imagem, mask=mask_inv)
        
        zeros = np.zeros(res.shape[:2], dtype = "uint8")
        test = cv2.bitwise_not(mask)

        #Divisão da area selecionada em canais RGB
        (canalAzul, canalVerde, canalVermelho) = cv2.split(res)

        #JUNTA O RGB E MUDA A COR - VARIA DE 0 A 255
        res = cv2.merge([canalAzul*0, canalVerde*0, canalVermelho*1])
        #Com a imagem do dado olhe esse parâmetro:
        #res = cv2.merge([canalAzul*0, canalVerde*255, canalVermelho*0])
        
        #Junta A mascara Invertida e a area REcolorida
        test = cv2.add(res,res1)
        #test = cv2.add(res,imagem)

        temp = np.vstack([
            np.hstack([imagem, res]),
            np.hstack([res1, test]),
            ])

        #temp1 = np.vstack([np.hstack([mask, mask_inv]),])
        
        cv2.imshow("RES", temp)
        #cv2.imshow("MN", temp1)

        cv2.waitKey(0)
        cv2.destroyAllWindows
        

#Instâncias
ulCor = ulColor()
ulCor1 = ulColor()
controle = Controle()
controle.main()
