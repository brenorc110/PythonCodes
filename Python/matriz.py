m=[[1,2,3],[4,5,6],[7,8,9]]
n=[[1,2,3],[4,5,6],[7,8,9]]
o=[]
# m= linha completa da matriz.

def somar(m,n):
	# Checar se  matrizes têm tamanhos iguais:
	if (len(m)== len(n)) and (len(m[0]) == len(n[0])):
		for i in range(len(m)):
			#Criar as linhas de O, senao elas nao existirao para as operacoes.	
			linha=[0]*len(m[0])
			o.append(linha)
			for j in range(len(m[0])):
				o[i][j] = m[i][j] + n[i][j]
	else:
		print("Matrizes nao tem mesmo tamanho. ")
	return o

#Cria matriz de zeros, para ser preenchida posteriormente pelo método de transpostas
def matriz_nula(nLinhas, nColunas):
	m0 = []
	for i in range (nLinhas):
		linha=[0]*nColunas
		m0.append(linha)
	return m0

def transposta(n):
	#print("Matriz transposta: ")
	nLinhas = len(n)
	nColunas = len(n[0])
	#Cria matriz nula com dimensoes transpostas
	t = matriz_nula(nColunas, nLinhas)
	for i  in range(nLinhas):
		for j in range(nColunas):
			t[j][i] = n[i][j]
	return t

def printMatriz(x):
	for i in x:
		#Pq j in i? Pq cada i recebe um elemento das linhas m.
		for j in i:
			print(j, end = ' ')
		print("\n")
"""
print("Matriz M: ")
printMatriz(m)
print("Matriz N: ")
printMatriz(n)
somar(m,n)
print("Matriz O: ")
printMatriz(o)
"""
print("Matriz N transposta: ")
printMatriz(transposta(n))