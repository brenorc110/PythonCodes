#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int i, k, j, SIZE=256, n;
time_t start,end;

float **A;
float **B;
float **C;

double **dA;
double **dB;
double **dC;

int alocaMatriz(int SIZE){
	//Aloca as linhas da matriz dinamicamente: 
	A = (float **) calloc (SIZE, sizeof(float *));
	B = (float **) calloc (SIZE, sizeof(float *));
	C = (float **) calloc (SIZE, sizeof(float *));
	
	//Aloca as colunas da matriz dinamicamente:
	for (i=0; i<SIZE; i++){
		A[i]= (float *) calloc (SIZE, sizeof(float));
		B[i]= (float *) calloc (SIZE, sizeof(float));
		C[i]= (float *) calloc (SIZE, sizeof(float));
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
	return 0;
}

int alocaMatrizDb(int SIZE){
	//Aloca as linhas da matriz dinamicamente: 
	dA = (double **) calloc (SIZE, sizeof(double *));
	dB = (double **) calloc (SIZE, sizeof(double *));
	dC = (double **) calloc (SIZE, sizeof(double *));
	
	//Aloca as colunas da matriz dinamicamente:
	for (i=0; i<SIZE; i++){
		dA[i]= (double *) calloc (SIZE, sizeof(double));
		dB[i]= (double *) calloc (SIZE, sizeof(double));
		dC[i]= (double *) calloc (SIZE, sizeof(double));
	}
	
	//inicializar matriz A
	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
			dA[i][j]=(2*i)+j;
		}
	}
	
	//inicializar matriz B
	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
			dB[i][j]=(3*i)-j;
		}
	}
	
	//inicializar matriz C
	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
			dC[i][j]=0;
		}
	}
	return 0;
}
		
int linear(int SIZE){
	start = clock();
	//Linear
	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
			for (k=0;k<SIZE;k++){
				C[i][j]=A[i][k]*B[k][j];
			}
		}
	}
	end = clock() - start;
	printf("Wall time FLOAT IJK LINEAR %i %fs \n", SIZE,((double)end)/CLOCKS_PER_SEC);
	return 0;
}

int unrowlling2(int SIZE){
	start = clock();
	//Unrowlling 2
	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
			for (k=0;k<SIZE;k=k+2){
				C[i][j]=A[i][k]*B[k][j];
				C[i][j]=A[i][k+1]*B[k+1][j];
			}
		}
	}
	end = clock() - start;
	printf("Wall time FLOAT IJK UNROWLLING 2 %i %fs \n", SIZE,((double)end)/CLOCKS_PER_SEC);
	return 0;
}

int unrowlling4 (int SIZE){
	start = clock();
	//Unrowlling 4
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
	end = clock() - start;
	printf("Wall time FLOAT IJK UNROWLLING 4 %i %fs \n", SIZE,((double)end)/CLOCKS_PER_SEC);
	return 0;
}

int blocking(int SIZE){
	start = clock();
	for (i=0;i<SIZE;i=i+2){
		for (j=0;j<SIZE;j=j+2){
			for (k=0;k<SIZE;k++){
				
				C[i][j]+=A[i][k]*B[k][j];	
			    C[i+1][j]+=A[i+1][k]*B[k][j]; 
			    C[i][j+1]+=A[i][k]*B[k][j+1];
			    C[i+1][j+1]+=A[i+1][k]*B[k][j+1];
			}
		}
	}

	end = clock() - start;
	printf("Wall time FLOAT IJK BLOCKING %i %fs \n", SIZE,((double)end)/CLOCKS_PER_SEC);
	system("pause");
	return 0;
}

int linearDb(int SIZE){
	start = clock();
	//Linear
	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
			for (k=0;k<SIZE;k++){
				dC[i][j]=dA[i][k]*dB[k][j];
			}
		}
	}
	end = clock() - start;
	printf("Wall time DOUBLE IJK LINEAR %i %fs \n", SIZE,((double)end)/CLOCKS_PER_SEC);
	return 0;
}

int unrowlling2Db(int SIZE){
	start = clock();
	//Unrowlling 2
	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
			for (k=0;k<SIZE;k=k+2){
				dC[i][j]=dA[i][k]*dB[k][j];
				dC[i][j]=dA[i][k+1]*dB[k+1][j];
			}
		}
	}
	end = clock() - start;
	printf("Wall time DOUBLE IJK UNROWLLING 2 %i %fs \n", SIZE,((double)end)/CLOCKS_PER_SEC);
	return 0;
}

int unrowlling4Db(int SIZE){
	start = clock();
	//Unrowlling 4
	for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
			for (k=0;k<SIZE;k=k+4){
				dC[i][j]=dA[i][k]*dB[k][j];
				dC[i][j]=dA[i][k+1]*dB[k+1][j];
				dC[i][j]=dA[i][k+2]*dB[k+2][j];
				dC[i][j]=dA[i][k+3]*dB[k+3][j];
			}
		}
	}
	end = clock() - start;
	printf("Wall time DOUBLE IJK UNROWLLING 4 %i %fs \n", SIZE,((double)end)/CLOCKS_PER_SEC);
	return 0;
}

int blockingDb(int SIZE){
	start = clock();
	for (i=0;i<SIZE;i=i+2){
		for (j=0;j<SIZE;j=j+2){
			for (k=0;k<SIZE;k++){
				C[i][j]+=A[i][k]*B[k][j];	
			    C[i+1][j]+=A[i+1][k]*B[k][j]; 
			    C[i][j+1]+=A[i][k]*B[k][j+1];
			    C[i+1][j+1]+=A[i+1][k]*B[k][j+1];
			}
		}
	}

	end = clock() - start;
	printf("Wall time DOUBLE IJK BLOCKING %i %fs \n", SIZE,((double)end)/CLOCKS_PER_SEC);
	system("pause");
	return 0;
}

int main(){
    for (n=0; n<4; n++){
        alocaMatriz(SIZE);
        linear(SIZE);
        unrowlling2(SIZE);
        unrowlling4(SIZE);
        blocking(SIZE);
	
	/*
        alocaMatrizDb(SIZE);	
	linearDb(SIZE);
	unrowlling2Db(SIZE);
	unrowlling4Db(SIZE);
	blockingDb(SIZE);
        SIZE =SIZE*2;
	*/
    }
	
	/*
	alocaMatrizFl(512);	
	linear(512);
	unrowlling2(512);
	unrowlling4(512);
	blockingFl(512);
		
	alocaMatrizDb(512);	
	linearDb(512);
	unrowlling2Db(512);
	unrowlling4Db(512);
	blockingDb(512);
	
	alocaMatrizFl(1024);	
	linear(1024);
	unrowlling2(1024);
	unrowlling4(1024);
	blockingFl(1024);
	
	alocaMatrizDb(1024);	
	linearDb(1024);
	unrowlling2Db(1024);
	unrowlling4Db(1024);
	blockingDb(1024);
	
	alocaMatrizFl(2048);	
	linear(2048);
	unrowlling2(2048);
	unrowlling4(2048);
	blockingFl(2048);

	alocaMatrizDb(2048);	
	linearDb(2048);
	unrowlling2Db(2048);
	unrowlling4Db(2048);
	blockingDb(2048);
	
	alocaMatrizFl(4096);	
	linear(4096);
	unrowlling2(4096);
	unrowlling4(4096);
	blockingFl(4096);
		
	alocaMatrizDb(4096);	
	linearDb(4096)
	unrowlling2Db(4096);
	unrowlling4Db(4096);
	blockingDb(4096);
	*/
	system("pause");
	return 0;
}
