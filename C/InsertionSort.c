	#include <stdio.h>
	#include <stdlib.h>
	
	void ordenacaoInsercao(int v[], int ni){
			int i,j, aux;
		for(i=1; i<ni;i++){
		
			aux=v[i];
			for(j=i;(j>0) && (aux<v[j-1]);j--)
			v[j]=v[j-1];
			v[j]=aux; 
		}
			
 for(i=0 ; i <5 ; i++)
printf("%da posicao do vetor ordenado: %d\n", i+1, v[i]); 
		
	}	
	
	void main(){
		int v[5];
		int i,n,aux;
		
		for(i=0 ; i <5 ; i++)
        {
            printf("Entre com o numero %d: \n", i+1);
            scanf("%d", &v[i]);
        }
 for(i=0 ; i <5 ; i++)
printf("%da posicao do vetor em main: %d\n", i+1, v[i]); 
		
		ordenacaoInsercao(v,5);
	}
