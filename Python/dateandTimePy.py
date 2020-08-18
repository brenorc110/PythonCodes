import datetime
currDate = datetime.date.today()
#Formatando a data, no caso exibindo no formato brasileiro
# Note que se usa %b para mês,, pq minutos também começa com m... 
print(currDate.strftime('%d %b, %Y'))

print(currDate.year)
print(currDate.month)
print(currDate.day)

# And it prints the whole time: 
currTime = datetime.datetime.now()
print(currTime)
print(currTime.hour)
print(currTime.minute)

# %H Hours(24hrs clock)
# %I Hours(12hrs clock)
# %p AM or PM
# %m Minutes
# %S Seconds

#timedelta allows you to specify the time to add or subtract from a date;
print(currDate + datetime.timedelta(days=15)

#Getting the difference between dates:
nextBirthday = \
    datetime.datetime.strptime('12/20/20141','%m/%d/%Y').date()
currDate = datetime.date.today()
difference = nextBirthday - CurrDate
print(difference.days)

#%d = Dia do mês.
#%b = Mês abreviado (Ex: Jan).
#%B = Mês completo.
#%y = Ano com dois dígitos.
#%Y = Ano completo.
#%a = Dia da semana abreviado.
#%A = Dia da semana completo.
# para mais, visite strftime.org
      
#DESPERTADOR:
from datetime import datetime
import time

now = datetime.now()
nowS = str(now)
nowSe = nowS.split(" ")
nowSeq = nowSe[1]
hora_atual = nowSeq[0:5]
hora = input("Informa a hora para despertar: HH:MM \n")

while hora_atual != hora: #A string é o horario requerido
  print("aguardando",hora_atual)
  now = datetime.now()
  nowS = str(now)
  nowSe = nowS.split(" ")
  nowSeq = nowSe[1]
  hora_atual = nowSeq[0:5]
  time.sleep(5)
else:
   print("Despertando!") #Saida quando atingido horario definido
# TESTE ALEMAAL!
