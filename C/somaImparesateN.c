#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int somaImpares(int n){
int i=0,soma=0;
i=0; 
while (i<=n){
if (i%2!=0)
soma=soma+i;
i++;
}
return soma;
}

int main(void){
int n;
printf("Entre com o numero: ");
scanf ("%d", &n);

printf ("Soma dos Impares: %d", somaImpares(n));
return 0;
}
