#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* run this program using the console pauser or add your own getch, system("pause") or input loop */
//int argc, char *argv[]
float calcula_area(float raio){
	float area;

	area = 3.14*pow(raio,2);

	return area;
}

void main() {
	float raio;
	
	printf("Escreva o raio: ");
	scanf ("%f", &raio);
	
	printf ("Raio %.2f\n", raio);
	
	if (raio <= 0)
		printf("Valor incorreto\n"); 
	else
	{
		
		printf("Area =  %.2f", calcula_area(raio));
		printf("\n");
	}
}


