#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
//windows
//gcc -fopenmp .\matrizesAndre.c -o matrizesOMP && matrizesOMP.exe
//g++ -fopenmp matrizesAndre.c -o matrizesAndre (EXECUTAR SOMENTE UMA VEZ, PRA GERAR O EXECUTAVEL).
//Agora simplesmente executa o exe: c:\Users\breno\Documents\Facul\APD\matrizesAndre.exe

//linux
//gcc -fopenmp multiplicacao-serial-Andre.c -o matrizesOMP && ./matrizesOMP

#include <omp.h>
#include <time.h>

int d1 = 256;
int d2 = 512;
int d3 = 1024;

void inicializaMatrizD1();
void inicializaMatrizD2();
void inicializaMatrizD3();

void *mythread1();  //thread prototype
void *mythread2();
void *mythread3();
void *mythread4();
void *mythread5();
void *mythread6();

double **A1;
double **B1;
double **C1;

double **A2;
double **B2;
double **C2;

double **A3;
double **B3;
double **C3;

float SerialKIJ(int dimensao, double **A , double **B, double **C) {
	
	int i, k, j;
	double t = omp_get_wtime();
	// #pragma omp for	
	for (k=0;k<dimensao;k++) {
		for (i=0;i<dimensao;i++) {
			for (j=0;j<dimensao;j++) {
				C[i][j]=A[i][k]*B[k][j];
			}
		}
	}	
	return omp_get_wtime() - t;
}

float SerialIKJ(int dimensao, double **A , double **B, double **C) {
	
	int i, k, j;
	double t = omp_get_wtime();	
	// #pragma omp for
	for (i=0;i<dimensao;i++) {
		for (k=0;k<dimensao;k++) {
			for (j=0;j<dimensao;j++) {
				C[i][j]=A[i][k]*B[k][j];
			}
		}
	}	
	return omp_get_wtime() - t;
}

float UnrollingX2KIJ(int dimensao, double **A , double **B, double **C) {
	int i, k, j;
	double t = omp_get_wtime();	
	// #pragma omp for
	for (k=0;k<dimensao;k++) {
		for (i=0;i<dimensao;i++) {
			for (j=0;j<dimensao;j=j+2) {
				C[i][j]=A[i][k]*B[k][j];
				C[i][j+1]=A[i][k]*B[k][j+1];
			}
		}
	}	
	return omp_get_wtime() - t;
}

float UnrollingX2IKJ(int dimensao, double **A , double **B, double **C) {
	int i, k, j;
	double t = omp_get_wtime();
	// #pragma omp for	
	for (i=0;i<dimensao;i++) {
		for (k=0;k<dimensao;k++) {
			for (j=0;j<dimensao;j=j+2) {
				C[i][j]=A[i][k]*B[k][j];
				C[i][j+1]=A[i][k]*B[k][j+1];
			}
		}
	}	
	return omp_get_wtime() - t;
}

float UnrollingX4KIJ(int dimensao, double **A , double **B, double **C) {
	int i, k, j;
	double t = omp_get_wtime();
	// #pragma omp for	
	for (k=0;k<dimensao;k++) {
		for (i=0;i<dimensao;i++) {
			for (j=0;j<dimensao;j=j+4) {
				C[i][j]=A[i][k]*B[k][j];
				C[i][j+1]=A[i][k]*B[k][j+1];
				C[i][j+2]=A[i][k]*B[k][j+2];
				C[i][j+3]=A[i][k]*B[k][j+3];
			}
		}
	}	
	return omp_get_wtime() - t;
}

float UnrollingX4IKJ(int dimensao, double **A , double **B, double **C) {
	int i, k, j;
	double t = omp_get_wtime();
	// #pragma omp for	
	for (i=0;i<dimensao;i++) {
		for (k=0;k<dimensao;k++) {
			for (j=0;j<dimensao;j=j+4) {
				C[i][j]=A[i][k]*B[k][j];
				C[i][j+1]=A[i][k]*B[k][j+1];
				C[i][j+2]=A[i][k]*B[k][j+2];
				C[i][j+3]=A[i][k]*B[k][j+3];
			}
		}
	}	
	return omp_get_wtime() - t;
}

float BlockingKIJ(int dimensao, double **A , double **B, double **C) {	
	//printf("\n");
	int i, k, j;
	double t = omp_get_wtime();
	// #pragma omp for
	for (k = 0; k < dimensao; k = k + 2) {
		for (i = 0; i < dimensao; i = i + 2) {
			for (j = 0; j < dimensao; j++) {			
				C[i][j] = A[i][k] * B[k][j];
				C[i][j] = A[i][k+1] * B[k+1][j];
				C[i+1][j] = A[i+1][k] * B[k][j];
				C[i+1][j] = A[i+1][k+1] * B[k+1][j];
			}
		}
	}
	return omp_get_wtime() - t;
}

float BlockingIKJ(int dimensao, double **A , double **B, double **C) {	
	int i, k, j;
	double t = 0;
	t = omp_get_wtime();	
	// #pragma omp for
	for (i = 0; i < dimensao; i = i + 2) {
		for (k = 0; k < dimensao; k = k + 2) {
			for (j = 0; j < dimensao; j++) {			
				C[i][j] = A[i][k] * B[k][j];
				C[i+1][j] = A[i+1][k] * B[k][j];
				C[i][j] = A[i][k+1] * B[k+1][j];
				C[i+1][j] = A[i+1][k+1] * B[k+1][j];
			}
		}
	}
	return omp_get_wtime() - t;
}

void printBlockingIKJ() {
	float tempo1 = BlockingIKJ(d1,A1,B1,C1);
	float tempo2 = BlockingIKJ(d2,A2,B2,C2);
	float tempo3 = BlockingIKJ(d3,A3,B3,C3);
	printf("Blockin IKJ ;  %.3f;  %.3f;  %.3f; \n", tempo1, tempo2, tempo3);	
}

void printBlockingKIJ() {
	float tempo1 = BlockingKIJ(d1,A1,B1,C1);
	float tempo2 = BlockingKIJ(d2,A2,B2,C2);
	float tempo3 = BlockingKIJ(d3,A3,B3,C3);
	printf("Blockin KIJ ;  %.3f;  %.3f;  %.3f; \n", tempo1, tempo2, tempo3);	
}

void printSerialIKJ() {
	float tempo1 = SerialIKJ(d1,A1,B1,C1);
	float tempo2 = SerialIKJ(d2,A2,B2,C2);
	float tempo3 = SerialIKJ(d3,A3,B3,C3);
	printf("Serial  IKJ ;  %.3f;  %.3f;  %.3f; \n", tempo1, tempo2, tempo3);	
}

void printSerialKIJ() {
	float tempo1 = SerialKIJ(d1,A1,B1,C1);
	float tempo2 = SerialKIJ(d2,A2,B2,C2);
	float tempo3 = SerialKIJ(d3,A3,B3,C3);
	printf("Serial  KIJ ;  %.3f;  %.3f;  %.3f; \n", tempo1, tempo2, tempo3);	
}

void printUnrollingX2IKJ() {
	float tempo1 = UnrollingX2IKJ(d1,A1,B1,C1);
	float tempo2 = UnrollingX2IKJ(d2,A2,B2,C2);
	float tempo3 = UnrollingX2IKJ(d3,A3,B3,C3);
	printf("Unrolx2 IKJ ;  %.3f;  %.3f;  %.3f; \n", tempo1, tempo2, tempo3);	
}

void printUnrollingX2KIJ() {
	float tempo1 = UnrollingX2KIJ(d1,A1,B1,C1);
	float tempo2 = UnrollingX2KIJ(d2,A2,B2,C2);
	float tempo3 = UnrollingX2KIJ(d3,A3,B3,C3);
	printf("Unrolx2 KIJ ;  %.3f;  %.3f;  %.3f; \n", tempo1, tempo2, tempo3);	
}

void printUnrollingX4IKJ() {
	float tempo1 = UnrollingX4IKJ(d1,A1,B1,C1);
	float tempo2 = UnrollingX4IKJ(d2,A2,B2,C2);
	float tempo3 = UnrollingX4IKJ(d3,A3,B3,C3);
	printf("Unrolx4 IKJ ;  %.3f;  %.3f;  %.3f; \n", tempo1, tempo2, tempo3);	
}

void printUnrollingX4KIJ() {
	float tempo1 = UnrollingX4KIJ(d1,A1,B1,C1);
	float tempo2 = UnrollingX4KIJ(d2,A2,B2,C2);
	float tempo3 = UnrollingX4KIJ(d3,A3,B3,C3);
	printf("Unrolx4 KIJ ;  %.3f;  %.3f;  %.3f; \n", tempo1, tempo2, tempo3);	
}

void comThreads(int numThreads) {

	printf("\nMETODO      ;   %d;   %d;   %d;\n",d1,d2,d3);	
	omp_set_num_threads(numThreads);
	double t = omp_get_wtime();
	#pragma omp parallel
	#pragma omp sections
	{
		#pragma omp section				
		printBlockingKIJ();
		#pragma omp section
		printBlockingIKJ();
		#pragma omp section
		printSerialKIJ();
		#pragma omp section
		printSerialIKJ();
		#pragma omp section
		printUnrollingX2KIJ();
		#pragma omp section
		printUnrollingX2IKJ();
		#pragma omp section
		printUnrollingX4KIJ();
		#pragma omp section
		printUnrollingX4IKJ();
	}
	printf("Tempo com %d thread: %.3f; \n",numThreads, omp_get_wtime() - t);
}

void semThreads() {

	double t = omp_get_wtime();
	printf("\nMETODO      ;   %d;   %d;   %d;\n",d1,d2,d3);	
	printBlockingKIJ();		
	printBlockingIKJ();		
	printSerialKIJ();		
	printSerialIKJ();		
	printUnrollingX2KIJ();		
	printUnrollingX2IKJ();		
	printUnrollingX4KIJ();		
	printUnrollingX4IKJ();
	printf("Tempo sem thread: %.3f; \n", omp_get_wtime() - t);
}

void comPThreads() {
	double t = omp_get_wtime();
	pthread_t thread[3];	
    //starting the thread
    pthread_create(&thread[0],NULL,mythread1,NULL);
    pthread_create(&thread[1],NULL,mythread2,NULL);
    pthread_create(&thread[2],NULL,mythread3,NULL);
    //waiting for completion
    pthread_join(thread[0],NULL);
    pthread_join(thread[1],NULL); 
    pthread_join(thread[2],NULL); 

	printf("Tempo com pthread: %.3f; \n", omp_get_wtime() - t);	
}

int main() {

	inicializaMatrizD1(d1);
	inicializaMatrizD2(d2);
	inicializaMatrizD3(d3);	

	semThreads();
	comThreads(2);
	comThreads(4);
	comThreads(8);
	comPThreads();
	

	return 0;
}

void inicializaMatrizD1(int dimensao) {

	int i, k, j;	
	int d1 = dimensao;
	A1 = (double **) calloc(d1, sizeof(double *));
	B1 = (double **) calloc(d1, sizeof(double *));
	C1 = (double **) calloc(d1, sizeof(double *));
	// #pragma omp for
	for ( i = 0; i < d1; i++ ) {
		A1[i] = (double *) calloc (d1, sizeof(double));
		B1[i] = (double *) calloc (d1, sizeof(double));
		C1[i] = (double *) calloc (d1, sizeof(double));
	}	
	// #pragma omp for
	for (i=0;i<d1;i++){
		for (j=0;j<d1;j++){
			A1[i][j]=(2*i)+j;
			B1[i][j]=(3*i)-j;
			C1[i][j]=0;
		}
	}
}

void inicializaMatrizD2(int dimensao) {

	int i, k, j;
	int d2 = dimensao;	
	A2 = (double **) calloc(d2, sizeof(double *));
	B2 = (double **) calloc(d2, sizeof(double *));
	C2 = (double **) calloc(d2, sizeof(double *));
	// #pragma omp for
	for ( i = 0; i < d2; i++ ) {
		A2[i] = (double *) calloc (d2, sizeof(double));
		B2[i] = (double *) calloc (d2, sizeof(double));
		C2[i] = (double *) calloc (d2, sizeof(double));
	}	
	// #pragma omp for
	for (i=0;i<d2;i++){
		for (j=0;j<d2;j++){
			A2[i][j]=(2*i)+j;
			B2[i][j]=(3*i)-j;
			C2[i][j]=0;
		}
	}
}

void inicializaMatrizD3(int dimensao) {

	int i, k, j;
	int d3 = dimensao;
	A3 = (double **) calloc(d3, sizeof(double *));
	B3 = (double **) calloc(d3, sizeof(double *));
	C3 = (double **) calloc(d3, sizeof(double *));
	// #pragma omp for
	for ( i = 0; i < d3; i++ ) {
		A3[i] = (double *) calloc (d3, sizeof(double));
		B3[i] = (double *) calloc (d3, sizeof(double));
		C3[i] = (double *) calloc (d3, sizeof(double));
	}	
	// #pragma omp for
	for (i=0;i<d3;i++){
		for (j=0;j<d3;j++){
			A3[i][j]=(2*i)+j;
			B3[i][j]=(3*i)-j;
			C3[i][j]=0;
		}
	}
}

//thread definition
void *mythread1(){
	printf("\n");
    printBlockingKIJ();
}

void *mythread2(){
	printf("\n");
	printBlockingIKJ();
}

void *mythread3(){
	printf("\n");
	printSerialKIJ();
}

void *mythread4(){
	printf("\n");
	printSerialIKJ();
}

void *mythread5(){
	printf("\n");
	printUnrollingX4KIJ();
}

void *mythread6(){
	printf("\n");
	printUnrollingX4IKJ();
}