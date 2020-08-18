#include <stdio.h>
#include <stdlib.h>

float paraMetrosPorSegundo(float v){
float ms = v/3.6;
return ms;
}

void main(void){
float v;
printf("Informe a vel. em KM/H: ");
scanf("%f", &v);

printf ("A vel. %.2f KMs/segundo em metros/segundo eh: %.2f", v,paraMetrosPorSegundo(v));
printf("\n");

}
