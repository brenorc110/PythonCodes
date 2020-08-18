from pybrain.structure import FeedForwardNetwork, BiasUnit, FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer

#Criando rede
rede = FeedForwardNetwork()

#Especificando camadas e neuronios: 
camadaEntrada = LinearLayer(2)
camadaOculta = SigmoidLayer(3)
camadaSaida= SigmoidLayer(1)

bias1 = BiasUnit()
bias2 = BiasUnit()

rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

entradaOculta = FullConnection(camadaEntrada, camadaOculta)
ocultaSaida = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2, camadaSaida)

rede.sortModules()

print (rede)
print (camadaOculta.params)
print (ocultaSaida.params)
print (biasOculta.params)
print (biasSaida.params)
