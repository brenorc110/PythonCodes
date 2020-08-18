import numpy as np
import cv2

imagem = cv2.imread('A.jpg')
imagem1 = cv2.imread('A.jpg')
hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
lower_green=np.array([20,20,20],np.uint8)           
upper_green=np.array([80,255,255],np.uint8)

mask = cv2.inRange(hsv, lower_green, upper_green)
res = cv2.bitwise_and(imagem,imagem, mask=mask)

mask_inv = cv2.bitwise_not(mask)

gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(gray, (7, 7), 0)

bordas = cv2.Canny(mask_inv, 5, 150)
(objetos, lx) = cv2.findContours(bordas.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

t2 = cv2.drawContours(imagem1, objetos, -1, (255, 0, 0), 1)



res1 = cv2.bitwise_and(imagem,imagem, mask=mask_inv)


(canalAzul, canalVerde, canalVermelho) = cv2.split(res)

zeros = np.zeros(res.shape[:2], dtype = "uint8")
test = cv2.bitwise_not(mask)

res = cv2.merge([zeros, canalVerde,zeros])

test = cv2.add(res,res1)

cv2.imshow("RES", test)
cv2.imshow("MN", mask)
cv2.imshow("MI", mask_inv)
      
cv2.imshow("Quantidade de objetos: ", t2)

#cv2.imshow("Vermelho", cv2.merge([zeros, zeros,canalVermelho]))
#cv2.imshow("Verde", cv2.merge([zeros, canalVerde, zeros]))
#cv2.imshow("Azul", cv2.merge([canalAzul, zeros, zeros]))

#Pegando as cores dos canais de acordo com porcentagem colocada. 1%B 0%G 90%RED (Só que no caso foi usado só o canal verde, que foi dividido em 3).
#cv2.imshow("Canal verde", cv2.merge([r * 0 , g * 0,  b * 0]))
# Agora devemos usar as demais cores, através de uma máscara invertida, ou seja, pegará tudo que é não-verde e complementará na imagem.
cv2.waitKey(0)
cv2.destroyAllWindows


gray = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)


