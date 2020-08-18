from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer, SigmoidLayer

#Especs. da rede neural 2 neuronios cada 1, 3 neuronios camada 2, 1 neuronio camada 3: 
# rede = buildNetwork(2, 3, 1, bias=False,
#                     hiddenclass=SigmoidLayer, outclass=Softmaxlayer)
# print (rede["in"])
# print (rede["hidden0"])
# print (rede["out"])
# print (rede["bias"])

rede = buildNetwork(2, 3, 1)

#Especifica 2 entradas que gerarão uma saída: 
base = SupervisedDataSet(2, 1)

#Add as entradas e a saída esperada, no caso, as do XOR: 
base.addSample((0, 0), (0, ))
base.addSample((0, 1), (1, ))
base.addSample((1, 0), (1, ))
base.addSample((1, 1), (0, ))

#Printa os dados de entrada: 
print(base('input'))

print(base('target'))
treinamento = BackpropTrainer(rede, dataset=base,
                              learningrate=0.01, momentum=0.06)

for x in range(0, 30000):
    erro = treinamento.train()
    if ((x % 1000) == 0):
        print ("Erro: " + str(erro))

print (rede.activate([0, 0]))
print (rede.activate([0, 1]))
print (rede.activate([1, 0]))
print (rede.activate([1, 1]))
