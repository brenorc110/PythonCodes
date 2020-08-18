# -*- coding: utf-8 -*-
"""
 Algoritmo XOR com aprendizado em redes neurais.
 Funcao que reajusta pesos:
"""
import numpy as np

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    #Funcao melhorada com derivadas (gradiente)
    return sig * (1 - sig)

#a = sigmoid(0.5)
#b = sigmoidDerivada(a)

#a = sigmoid(-1.5)
#b = np.exp(0)

entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])

saidas = np.array([[0],[1],[1],[0]])
#Pesos da camada de entrada para a oculta: 
pesos0 = np.array([[-0.424, -0.740, -0.961],
                   [0.358, -0.577, -0.469]])

#Pesos da camada oculta pra camada de saida:
pesos1 = np.array([[-0.017], [-0.893], [0.148]])

#Gerando pesos aleatórios: 2,3 pq sao 2 neuronios na entrada, e 3 na oculta.
#pesos0 = np.random.random((2,3))
#pesos1 = np.random.random((3,1))

epocas = 1000000
taxaAprendizagem = 0.3
momento = 1

for j in range(epocas):
    print("Epoca atual:", j)
    camadaEntrada = entradas
    #dot = produto escalar = pega as entradas e soma à multiplicação dos pesos.
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro: " + str(mediaAbsoluta))
    #===============FEEDFORWARD e Delta============================
    #Calculo delta camada de saida:
    #DeltaSaida = Erro * sigmoidDerivada
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    #Delta camada oculta (Funcao = sigmoidDerivada * peso * DeltaSaida)
    #A seguir transpomos os pesos (colunas viram linhas), para multiplicacao em matriz
    #Pois do contrario dará erro, pois nao é possivel multiplicar 4 valores por 3 em linha.
    pesos1Transposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    #DeltaEscondida * sigmoidDerivada:
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    #===============BACKPROPAGATION=================================
    #Ele reajusta os pesos com base na formula abaixo, mas, da última pra primeira camada, daí o nome.
    #Peso n+1 = (peso n * momento) + (entrada * DeltaSaida * taxaAprendizagem)
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    #print("Pesos reajustados FINAL PRA OCULTA= " + str(pesosNovo1))
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    #print("Pesos reajustados OCULTA PRA ENTRADA= " + str(pesosNovo0))
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)