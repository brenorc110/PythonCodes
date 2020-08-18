import turtle
#Quatro quadrados em 1 - versão 1:
#for stepes in range (4):
#    turtle.forward(100)
#    turtle.right(90)

#for steps2 in range (3):
#    turtle.forward(100)
#    turtle.left(90)

#for steps3 in range (4):
#    turtle.forward(100)
#    turtle.right(90)

#turtle.forward(200)
#turtle.right(90)
#turtle.forward(100)
#turtle.right(90)
#turtle.forward(100)
#turtle.right(90)
#turtle.forward(100)
#turtle.left(45)
#turtle.forward(100)

#Quatro quadrados em 1 com SUBLOOP:
#for steps in range (4):
#    turtle.forward(100)
#    turtle.right(90)
#    for steps2 in range(4):
#        turtle.forward(50)
#        turtle.right(90)

# 2ND PART: 
#Desenha qualquer figura especificando os lados:
#nbrSides = 50
#for stesps in range(nbrSides):
#    turtle.forward(50)
#   turtle.right(360/nbrSides)
#    for moresteps in range (nbrSides):
#        turtle.forward(25)
#        turtle.right(360/nbrSides)

# SUBSQUARES (LONG CODE):
#for i in range (4):
#    turtle.forward(100)
#    turtle.right(90)
#turtle.right(45)
#turtle.forward(10)
#turtle.left(45)
#for i in range (4):
#    turtle.forward(85)
#    turtle.right(90)
#turtle.right(45)
#turtle.forward(10)
#turtle.left(45)
#for i in range (4):
#    turtle.forward(75)
#    turtle.right(90)
#turtle.right(45)
#turtle.forward(10)
#turtle.left(45)
#for i in range (4):
#    turtle.forward(65)
#    turtle.right(90)
#turtle.right(45)
#turtle.forward(10)
#turtle.left(45)
#for i in range (4):
#    turtle.forward(55)
#    turtle.right(90)
#turtle.right(45)
#turtle.forward(10)
#turtle.left(45)
#for i in range (4):
#    turtle.forward(45)
#    turtle.right(90)
#turtle.right(45)
#turtle.forward(10)
#turtle.left(45)
#for i in range (4):
#    turtle.forward(35)
#    turtle.right(90)
#turtle.right(45)
#turtle.forward(10)
#turtle.left(45)
#for i in range (4):
#    turtle.forward(25)
#    turtle.right(90)
#turtle.right(45)
#turtle.forward(10)
#turtle.left(45)
#for i in range (4):
#    turtle.forward(15)
#    turtle.right(90)
#turtle.right(45)
#turtle.forward(7)
#turtle.left(45)
#for i in range (4):
#    turtle.forward(7)
#    turtle.right(90)

#SUBSQUARES (OPTIMIZED CODE):
for x in range (100, 0, -10):
    for i in range (4):
        
        turtle.forward(x)
        turtle.right(90)

    turtle.right(45)
    turtle.forward(10)
    turtle.left(45)
    

