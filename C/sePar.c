#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int ehPar(int n){
if (n%2==0)
return 1;
else
return 0; 
}
void main(void){
int n;
printf("Entre com o numero: ");
scanf ("%d", &n);

printf ("Resultado= %d", ehPar(n));
printf ("\nOBS: \n");
printf ("1 = numero Par\n"); 
printf ("0 = numero Impar\n"); 
}
