#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]){
	float C, F;
	printf ("Digite a temperatura em Celsius: ");
	scanf("%f", &C);
	F = (9*C+160)/5;
	printf ("Esta temperatura corresponde a %.1f graus F\n", F);
	return 0;
}
