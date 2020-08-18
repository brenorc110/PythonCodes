import sys
class Agenda(object):
    def __init__(self, opcion=0, continua="s", agenda={}, lista=[]):
        self.opcion = opcion
        self.continua = continua
        self.agenda = agenda
        self.lista = lista
    def cargar(self, continua, agenda):
        
        fecha=input("Ingrese la fecha de la actividad:")

        while continua=="s":
            hora=input("Ingrese la hora de la actividad con formato hh:mm \n")
            actividad=input("Ingrese la descripcon de la actividad:")
            self.lista.append((hora, actividad))
            agenda[fecha] = self.lista
            
            continua=input("Ingresa otra actividad para la misma fecha[s/n]: \n")
            if continua =="n":
                continua2=input("Ingresar entonces en otra fecha? [s/n]:")
                if continua2 == "s":
                    fecha=input("Ingrese la fecha de la actividad:")
                    continua="s"
                else:
                    op_menu = input("Quieres volver al menú principal? ")
                    if op_menu == "s":
                        ea.main()
                    else:
                        sys.exit()

    def consulta_fecha(self, continua):
        while self.continua=="s":
            fecha=input("ingrese la fecha con formato dd/mm/aa: \n")
            if fecha in agenda:
                for hora,actividad in agenda[fecha]:
                    print(hora,actividad)
            else:
                print("No hay actividades agendadas para dicha fecha \n")
                opcion2 = input("Quiere hacer otra consulta de fecha? \n")
                if opcion2=="s":
                    consulta_fecha()
                else:
                    self.continua="n"
                    
    def imprimir(self, agenda):
        print("Listado completo de la agenda \n")
        for fecha in agenda:
            #print("Para la fecha: ", fecha)
            for hora, actividad in agenda[fecha]:
                print(fecha, hora,actividad)
            
    def main(self):
        opcion = input("Escolha sua opção: " \
                       " 1 - Inserir actividad."\
                       " 2 - Consultar actividad por fecha. "\
                       " 3 - Imprimir agenda." \
                       " o 4 para SALIR\n")
        if opcion == "1":
            ea.cargar("s", self.agenda)
        elif opcion == "2":
            ea.consulta_fecha("s")
        elif opcion == "3":
           ea.imprimir(self.agenda)
        elif opcion == "4":
            sys.exit()
        else:
            print("Opcion incorrecta!")

ea = Agenda()
ea.main()
