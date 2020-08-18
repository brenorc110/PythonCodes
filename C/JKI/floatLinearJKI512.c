#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define SIZE 512

int main(){
	
	int i, k, j;
	time_t start,end;
	
	float **A;
	float **B;
	float **C;
	
	//Aloca as linhas da matriz dinamicamente: 
	A = (float **) calloc (SIZE, sizeof(float *));
	B = (float **) calloc (SIZE, sizeof(float *));
	C = (float **) calloc (SIZE, sizeof(float *));
	
	//Aloca as colunas da matrioz dinamicamente:
	for (i=0; i<SIZE; i++){
		A[i]= (float *) calloc (SIZE, sizeof(float));
		B[i]= (float *) calloc (SIZE, sizeof(float));
		C[i]= (float *) calloc (SIZE, sizeof(float));
	}
	
	start = clock();
	//inicializar matriz A
	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
			A[i][j]=(2*i)+j;
		}
	}
	
	//inicializar matriz B
	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
			B[i][j]=(3*i)-j;
		}
	}
	
	//inicializar matriz C
	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
			C[i][j]=0;
		}
	}
	
	//multiplicacao de A por b
	
	for (j=0;j<SIZE;j++){
		for (k=0;k<SIZE;k++){
			for (i=0;i<SIZE;i++){
				C[i][j]=A[i][k]*B[k][j];
			}
		}
	}
	
	/*
	for (j=0;j<SIZE;j++){
		for (k=0;k<SIZE;k++){
			for (i=0;i<SIZE;i=i+2){
				C[i][j]=A[i][k]*B[k][j];
				C[i+1][j]=A[i+1][k]*B[k][j];
			}
		}
	}

	for (j=0;j<SIZE;j++){
		for (k=0;k<SIZE;k++){
			for (i=0;i<SIZE;i=i+4){
				C[i][j]=A[i][k]*B[k][j];
				C[i+1][j]=A[i+1][k]*B[k][j];
				C[i+2][j]=A[i+2][k]*B[k][j];
				C[i+3][j]=A[i+3][k]*B[k][j];
			}
		}
	}
	*/
	end = clock() - start;	
	printf("Wall time: %lf s\n", ((double)end)/CLOCKS_PER_SEC);	
	//system("pause");
	return 0;
}
