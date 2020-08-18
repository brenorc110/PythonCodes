###Python program to print Fizz Buzz 
###loop for 100 times i.e. range 
##for fizzbuzz in range(101):  
##  
##    # number divisible by both 2 and 3, print 'FizzBuzz'  
##    # in place of the number 
##    if fizzbuzz % 2 == 0 and fizzbuzz % 3 == 0:  
##        print("FizzBuzz")                                          
##        continue
##  
##    # number divisible by 3, print 'Fizz' 
##    # in place of the number 
##    elif fizzbuzz % 3 == 0:      
##        print("Fizz")                                          
##        continue
##  
##    # number divisible by 2, print 'Buzz' in 
##    # place of the number 
##    elif fizzbuzz % 2 == 0:          
##        print("Buzz")                                      
##        continue
##  
##    # print numbers 
##    print(fizzbuzz) 


from functools import reduce
print(reduce(lambda x,y: x + "\n"+ y, map(lambda x: "Fizz" * ( x % 3 == 0) + "Buzz" *( x % 2 == 0) or str(x), range(101))))
