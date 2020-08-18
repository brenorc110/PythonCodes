#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* run this program using the console pauser or add your own getch, system("pause") or input loop */
float rendePoupanca(float poupanca){
	float novoValor;
	float juros =  1.01;
	novoValor = poupanca*juros;
	return novoValor; 
}

void main(void){
	float poupanca;
	float juros;
	poupanca = 500.0;
	poupanca = rendePoupanca(poupanca);
	poupanca = poupanca+200;
	poupanca = rendePoupanca(poupanca);
	poupanca = poupanca-50;
	poupanca = rendePoupanca(poupanca);
	
	printf("Eu tenho %f reais na poupanca. ", poupanca);
	printf("\n");	
}

