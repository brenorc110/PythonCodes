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
	start = clock();	
	
	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
				for (k=0;k<SIZE;k++){
				C[i][j]=A[i][k]*B[k][j];
			}
		}
	}
	
	/*
	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
				for (k=0;k<SIZE;k=k+2){
				C[i][j]=A[i][k]*B[k][j];
				C[i][j]=A[i][k+1]*B[k+1][j];
			}
		}
	}

	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
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
	system("pause");
	return 0;
}
