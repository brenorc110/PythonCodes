/*#include <stdio.h>
#include <stdlib.h>*/

char *pilha;
int topo;

//inicializa pilha
void inicializa_pilha(int n)
{
    pilha = malloc(n * sizeof(char));
    topo = 0;
}

//verifica se pilha está vazia
int empty()
{
    return (topo==0);
}

//empilhar
void push(char exp[])
{
    pilha[topo++] = exp;
}

//desempilhar
char pop()
{
    return pilha[--topo];
}

//retorna topo
char peek()
{
    return pilha[topo-1];
}

//exclui todos elementos da pilha
void clear()
{
    free (pilha);
    pilha = NULL;
}
char varrePilha(){
	if(pilha[topo] == ")" && pilha[topo-1] == "(") pop();

	return pilha[topo-1];
}

int main( int args, char * arg[]){
int tamanho;
char c[];

	void inicializa_pilha(5);

for(i=0; i=4; i++){
	printf("Entre com o elemento:\n"); 
	scanf("%c", &c[i]);
	
	push(c[i]);
	
}

	tamanho = strlen (c[]);
	printf("Numero total de letras:\n %d", tamanho);
	varrePilha();
		
	if (pilha[topo]==0){
		printf("Estrutura correta!");
	} else {
		printf("Estrutura incorreta!");
	}
	return 0; 
}