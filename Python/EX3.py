n =int(input("Entre com o número de elementos de X: "))

X = [int (input("Digite o {0}{1}: ".format(i+1,chr(176))))for i in range (n)]
print("O vetor de entrada é: ")
print(X)

def insertsort(X):
  for i in range (1, n):
    eleito = X[i]
    j = i -1
    while j >=0 and X[j] > eleito:
      X[j+1] = X[j]
      j = j-1
      
    X[j+1] = eleito

insertsort(X)
print("O vetor ordenado é: ")
print(X)