def main():
    names = getNames()
    printNames(names)
    return

def getNames():
    names = ['Anna', 'Brenno']
    name = input('Enter next name, pls: ')
    names.append(name)
    return names

def printNames(names):
    for name in names:
        print(name)
    return

main() #execute the main function

#---FACT
#def fatorial(n):
#   fact = 1
#    while n > 1:
#        fact = fact * n
#        n = n - 1
#    return fact

#print("Fatorial: ", fatorial(6))

def numprimo(num):
    for n in range(2, num):
        if num %n==0:
        print('Is not primo.')
        else:
            print('Primo')

#Expressões LAMBDA:
def square(num):
    return num**2

square=lambda num: num**2

par = lambda x: x%2==0
par(2)

inverter_string = lambda s: s[::-1]
inverter_string('Olá mundo!')

#Sequencia de Fibonacci (a[3] = a[2]+a[1]) com GERADORES: 
def genfib(n):
    a=1
    b=1
    for i in range(n):
        yield a
        a, b =b, a + b
                
for n in genfib(10):
    print(n)

