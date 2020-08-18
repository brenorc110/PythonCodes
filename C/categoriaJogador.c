#include <stdio.h>
#include <stdlib.h>


void categoria (int idade, float altura){
	if (idade >=18 && altura> 1.90)
		printf ("Voce esta fora da categoria A! \n");
	 else 
	 printf ("Voce esta dentro \n");
}


void main(){
	int idade;
	float altura;
	
	printf ("Escreva a sua idade: ");
	scanf ("%d", &idade);
	
	printf ("Escreva a sua altura: ");
	 scanf ("%f", &altura);
	
categoria(idade, altura);
		printf("\n");
}
