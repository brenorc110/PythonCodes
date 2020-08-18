#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define SIZE 1024

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
		for (i=0;i<SIZE;i++){
			for (k=0;k<SIZE;k++){
				C[i][j]=A[i][k]*B[k][j];
			}
		}
	}
	/*
	for (j=0;j<SIZE;j++){
		for (i=0;i<SIZE;i++){
			for (k=0;k<SIZE;k=k+2){
				C[i][j]=A[i][k]*B[k][j];
				C[i][j]=A[i][k+1]*B[k+1][j];
			}
		}
	}

	for (j=0;j<SIZE;j++){
		for (i=0;i<SIZE;i++){
			for (k=0;k<SIZE;k=k+4){
				C[i][j]=A[i][k]*B[k][j];
				C[i][j]=A[i][k+1]*B[k+1][j];
				C[i][j]=A[i][k+2]*B[k+2][j];
				C[i][j]=A[i][k+3]*B[k+3][j];
			}
		}
	}
	*/
	end = clock() - start;	
	
	printf("Wall time: %lf s\n", ((double)end)/CLOCKS_PER_SEC);	
	
	//system("pause");
	return 0;
}
