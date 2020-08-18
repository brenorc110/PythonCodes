#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int ehDivisivelPor3ou5(int n){
if (n%3==0 && n%5==0)
return 0;
else if (n%3==0 || n%5==0)
return 1;
else
return 0;
}
int main(){
int n;
printf ("Entre com o número: ");
scanf ("%d", &n);

printf ("Resultado= %d", ehDivisivelPor3ou5(n));
printf ("\nOBS: \n");
printf ("1 = Numero divisivel por 3 ou 5. \n"); 
printf ("0 = Divisivel pelo dois ou por nenhum. ");
return 0; 
}

