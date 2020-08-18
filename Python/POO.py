class Shark:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def swim(self):
        print(self.name + " is swimming like a shark.")
    def be_awesome(self):
        print(self.name + " is being awesome...")
    def __swim_backwards(self):
        print("Chanfle! Nothing can swim backwards!")
        
def main():
    sammy = Shark("Sammy", 10)
    sammy.be_awesome()
    stevie = Shark("Stevie", 11)
    stevie.swim()
    print("They're respectly", sammy.age, "and",  stevie.age, "Y.O.")

if __name__ == "__main__":
    main()
    
class WhaleShark(Shark):
    #OBS: Se inicializar algo aqui vai dar chabu! XD
    pass
        
terry = WhaleShark("Terry", 2)
print("Nome: " + terry.name)
terry.swim()

class WhiteShark(Shark):
    
    def live_with_anemone(self):
        print("The Whiteshark is coexisting with sea anemone.")

casey = WhiteShark("Casey", 2)
print("Nome: " + casey.name)
casey.swim()
casey.live_with_anemone()
#casey.__swim_backwards()

class Hammerhead(Shark):
    def __init__(self, water="freshwater"):
        self.water = water
        super().__init__(self,2)
angus = Hammerhead()
angus.name= "Angus"

print("It's called: " + angus.name)
print("It's a " + angus.water + " fish.")

#MULTIPLE HERITAGE:
#Let's add another class:
class Fish:
    def __Init__(self, name, specie):
        self.name = name
        self.specie = specie
        
    def fish_swim(self):
        print(self.name + " is swimming like a fish.")

class NurseShark(Shark, Fish):
    pass

angela = NurseShark("Angela", 11)
print (angela.name)
angela.swim()
angela.fish_swim()
#Inherited 

#AND MORE STUFF! 
##class Circle(object):
##    pi = 3.14
##
##    # O círculo é instanciado com um raio (o padrão é 1)
##    def __init__(self, radius=1):
##        self.radius = radius 
##
##    # Método de cálculo da área. Observe o uso de si mesmo.
##    def area(self):
##        return self.radius * self.radius * Circle.pi
##
##    # Método que redefine a área
##    def setRadius(self, radius):
##        self.radius = radius
##
##    # Método para obter raio (Mesmo que apenas chamar .radius)
##    def getRadius(self):
##        return self.radius
##
##
##c = Circle()
##
##c.setRadius(2)
##print('O raio é :',c.getRadius())
##print('A área é',c.area())
##
##class Book(object):
##    def __init__(self, title, author, pages):
##        print("A book is created")
##        self.title = title
##        self.author = author
##        self.pages = pages
##
##    def __str__(self):
##        return "Title:%s , author:%s, pages:%s " %(self.title, self.author, self.pages)
##
##    def __len__(self):
##        return self.pages
##    #Classe destrutora do objeto
##    def __del__(self):
##        print("A book is destroyed")
##
##book = Book("Python Rocks!", "Rodrigo Tadewald", 159)
##
### Métodos especiais
##print(book)
##print(len(book))
##del book

