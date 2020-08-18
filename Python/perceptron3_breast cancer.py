import numpy as np
from sklearn import datasets

def sigmoid(soma):
    return (1 / (1 + np.exp(-soma)))

def sigmoidDerivada(sig):
    return (sig * (1 - sig))

#Le base de dados e formata as saídas.
base = datasets.load_breast_cancer()
entradas = base.data
valoresSaida = base.target
saidas = np.empty([569, 1], dtype=int)

#Jogando as saidas para um array
for a in range(569):
    saidas[a] = valoresSaida[a]

pesos0 = (2 * (np.random.random((30, 3)) - 1)) #Num de dados de entrada, numero de neuronios da camada escondida
pesos1 = (2 * (np.random.random((3, 1)) - 1))

epocas = 1000000
taxaAprendizagem = 0.3
momentum = 1

for x in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)

    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)

    erroCamadaSaida = (saidas - camadaSaida)
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))

    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = (erroCamadaSaida * derivadaSaida)

    pesos1Transposta = pesos1.T
    deltaSaidaXpeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = (deltaSaidaXpeso * sigmoidDerivada(camadaOculta))

    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = ((pesos1 * momentum) + (pesosNovo1 * taxaAprendizagem))

    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = ((pesos0 * momentum) + (pesosNovo0 * taxaAprendizagem))

    print ("Error: " + str(mediaAbsoluta))
    print ("Pesos 0: ", pesos0)
    print ("Pesos 1: ", pesos1)
