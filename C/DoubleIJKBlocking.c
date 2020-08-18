#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
int SIZE = 2048;

	int i, k, j, n;
	time_t start,end;
	
	double **A;
	double **B;
	double **C;
	
int execucaoSerial(int SIZE){
    	//multiplicacao de A por b
	start = clock();
	
	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
				for (k=0;k<SIZE;k++){
				C[i][j]=A[i][k]*B[k][j];
			}
		}
	}
	end = clock() - start;	
	printf("Wall time DOUBLE IJK LINEAR: %lf s\n", ((double)end)/CLOCKS_PER_SEC);	
	return 0;
}
int execucaoBlocking(int SIZE){
    start=clock();
	for (i=0;i<SIZE;i=i+2){
		for (j=0;j<SIZE;j=j+2){
			for (k=0;k<SIZE;k++){
				
				C[i][j]=C[i][j] + A[i][k]*B[k][j];	
			    C[i+1][j]=C[i+1][j] + A[i+1][k]*B[k][j]; 
			    C[i][j+1]=C[i][j+1] + A[i][k]*B[k][j+1];
			    C[i+1][j+1]=C[i+1][j+1] + A[i+1][k]*B[k][j+1];
			}
		}
	}

	end = clock() - start;	
	printf("Wall time DOUBLE IJK BLOCKING: %lf s\n", ((double)end)/CLOCKS_PER_SEC);	
}

int main(){

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
	for (n=0; n<2; n++){
	    execucaoSerial(SIZE);
	    execucaoBlocking(SIZE);
		SIZE=SIZE*2
	}

	system("pause");
	return 0;
}
