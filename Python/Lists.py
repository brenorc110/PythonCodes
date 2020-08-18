#guest=['Christopher','Susan']
#print(guest[0])

## To print the last position, just use print(guest[-1])!!!
## It would be great in a bigger list

#guest.append('Brenno')
#print(guest)

#guest.remove('Brenno')
#del guest[0]
#print(guest[0])

##Get the index of an item.
#guest.append('Susan')
#print('Susan is in the position', guest.index('Susan'))
#guest.append('Anna')
#print('The updated list is', guest)

#posicoes = len(guest)

#for i in range(posicoes):
#    print(guest[i])
#    i+=1

## Code for adding guests till DONE was inserted:

#guest=[]
#name=" "

#while name.upper() != "DONE":
#    name=input("Enter guest name, or DONE to finish:")
#    guest.append(name)

#guest.sort()
#for guest in guest:
#    print(guest)

##---Mais dicas - lista---
#print("Palavras em uma lista:")
#for palavra in ["Algoritmos", "em", "Python"]:
#    print(palavra)

#quadrados = [n ** 2 for n in range(0, 10)]
#print(quadrados)

#quadrados_numeros_impares = [n ** 2 for n in range(0, 10) if n % 2 == 1]
#print(quadrados_numeros_impares)

# def encontra_elementos_duplicados(lista, m):
    # """
    # Imprime os números que aparecem mais de uma vez na lista de entrada.
    # É garantido que todos os números na lista de entrada estão no intervalo [0, m].
    # """
    #Retorna zero se a lista de entrada estiver vazia.
    # if not lista: 
        # return []
    
    #Procura por elementos repetidos na lista.
    # duplicatas = []
    # for i in range(len(lista)):
        # for j in range(i + 1, len(lista)):
            # if lista[i] == lista[j]:
                # duplicatas.append(lista[j])
    
    #A saída do algoritmo é a lista de elementos repetidos.
    # return duplicatas

# Matrizes (conjunto de listas): 
#list_1=[1,2,3]
#list_2=[4,5,6]
#list_3=[7,8,9]
#matriz= [list_1, list_2, list_3]
#for i in range (3):
#    for j in range (3):
#        temp = matriz[i][j]
#        temp = pow(temp,2)
#        matriz[i][j] = temp
#        print(matriz[i][j])
#       temp=0

# pra pegar qualquer posiçao da matriz, [indice][indice], primeiro índice referente à lista, (0, 1 e 2) e depois à posição.
#matriz[2][1] # Pega o número 9.

#SET é um tipo de conjunto que não aceita duplicatas, ou seja, ele unifica os elementos.
#x= set(matriz)
#x.add(1,2,2,3)

#Nesse caso só add os números pares até 10 à lista:
#x= [i for i in range (0,10) if i%2==0]

#lista=[]
#lista=[letter for letter in 'Asasbanderhousenhozenflausen']

#string= 'String aqui here hier'
#for char in string:
#    print(char)

# Crio uma cópia temporária e converto na nova
celsius=[0,10,15,20,30,50,100]
farenheit=[(temp*(9/5)+32) for temp in celsius]
