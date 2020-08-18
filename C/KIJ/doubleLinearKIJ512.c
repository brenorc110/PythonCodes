#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define SIZE 512

int main(){
	
	int i, k, j;
	time_t start,end;
	
	double **A;
	double **B;
	double **C;
	
	//Aloca as linhas da matriz dinamicamente: 
	A = (double **) calloc (SIZE, sizeof(double *));
	B = (double **) calloc (SIZE, sizeof(double *));
	C = (double **) calloc (SIZE, sizeof(double *));
	
	//Aloca as colunas da matriz dinamicamente:
	for (i=0; i<SIZE; i++){
		A[i]= (double *) calloc (SIZE, sizeof(double));
		B[i]= (double *) calloc (SIZE, sizeof(double));
		C[i]= (double *) calloc (SIZE, sizeof(double));
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
	
	for (k=0;k<SIZE;k++){
		for (i=0;i<SIZE;i++){
			for (j=0;j<SIZE;j++){
				C[i][j]=A[i][k]*B[k][j];
			}
		}
	}
	
	/*
	for (k=0;k<SIZE;k++){
		for (i=0;i<SIZE;i++){
			for (j=0;j<SIZE;j=j+2){
				C[i][j]=A[i][k]*B[k][j];
				C[i][j+1]=A[i][k]*B[k][j+1];
			}
		}
	}
	
	for (k=0;k<SIZE;k++){
		for (i=0;i<SIZE;i++){
			for (j=0;j<SIZE;j=j+4){
				C[i][j]=A[i][k]*B[k][j];
				C[i][j+1]=A[i][k]*B[k][j+1];
				C[i][j+2]=A[i][k]*B[k][j+2];
				C[i][j+3]=A[i][k]*B[k][j+3];
			}
		}
	}
	*/
	end = clock() - start;	
	printf("Wall time: %lf s\n", ((double)end)/CLOCKS_PER_SEC);	
	system("pause");
	return 0;
}
