#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int fatorial(int n){
int fat;
for(fat = 1; n > 1; n = n - 1)
fat = fat * n;	
}
int main()
{
int n;
printf("Insira um valor para o qual deseja calcular seu fatorial: ");
scanf("%d", &n);
 
printf("\nFatorial calculado: %d", fatorial(n));
return 0; 
}
