fileName = 'demo.txt'
WRITE = 'w'
APPEND = 'a'

data = input('Enter file INFO: ')
file = open(fileName, mode=WRITE)
file.write(data)
file.close()

##file=open('fileName', mode = WRITE)
#file.write('Breno, 23 \n')
#file.write('Sandra, 43 \n')
#file.close()

print('File writen successfully. Dake sehr!')

file.read()
file.seek() # pra ele voltar o cursor pro começo, e
file.readline() # Ler a linha atual. 