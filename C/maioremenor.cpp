#include <stdio.h>
#include <stdlib.h>

/* run this program using the console pauser or add your own getch, system("pause") or input loop */

int main(int argc, char *argv[]) {
int a,b,c,maior,menor; 

 printf("Entre o primeiro numero: \n");
 scanf("%d",&a); 
 
 printf("Entre o segundo numero: \n");
 scanf("%d",&b); 
 
 printf("Entre o terceiro numero: \n");
 scanf("%d",&c); 
 
 printf("O A eh: %d \n", a); 
  
  
 if (a>b){
 	maior = a;
 	menor = b;
 } else {
 	menor = a;
 	maior = b;
 }
 if(c>maior) maior = c;
 else if (c<menor) menor = c;
 
 printf("O maior numero eh: %d \n", maior);
 printf("O menor numero eh: %d \n", menor);
 

return 0;
}
