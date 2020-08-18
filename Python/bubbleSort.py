#usr/bin/env python
# -*- coding: utf-8 -*-

n = int(input("Entre com o número de elementos de X: "))

#Criando a lista de 5 elementos a partir da entrada do usuário
X = [int(input("Digite o {0}{1}: ".format(i+1, chr(176)))) for i in range(n)]
print("O vetor de entrada é:  ")
print(X)
#Ordenação de forma crescente, a partir do número de elementos em X

#Laço com a quantidade de elementos do vetor
for k in range(1, n+1):
  #Laço que percorre da primeira à penultima posição do vetor
  for i in range(0, n-1):    
    if(X[i] > X[i+1]):
      X[i],X[i+1]=X[i+1],X[i]
print("O vetor ordenado é: ")
print(X)
