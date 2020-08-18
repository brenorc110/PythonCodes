#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* run this program using the console pauser or add your own getch, system("pause") or input loop */
float calculaIMC(float peso, float altura){
	float imc;
	imc=peso/pow(altura,2);
	return imc;
}
void main() {
float peso, altura;

	printf("Escreva o peso: ");
	scanf ("%f", &peso);
	
	printf("Escreva a altura: ");
		scanf ("%f", &altura);
	
	printf("Seu IMC eh %.2f\n", calculaIMC(peso, altura));
	printf("Abaixo de 18.5 = abaixo do peso.\n");
	printf("Entre 18.6 e 24.9 = dentro do peso.\n");
	printf("Entre 25 e 29.9 = Um pouco acima do peso.\n");
	printf("Entre 30 e 34.9 = Obesidade grau 1.\n");
	printf("Entre 35 e 39,9 = Obesidade grau 2.\n");
	printf("Acima de 40 = Obesidade morbida.\n");
	printf("\n"); 
}
