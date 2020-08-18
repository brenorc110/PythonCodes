int fatorial (int n)
{
	if(n==0)
	return 1;
	else
	return n * fatorial(n-1);
}


int main (){
	int n;
	printf("Insira um valor para o qual deseja calcular seu fatorial: ");
	scanf("%d", &n);
 
	printf("\nFatorial calculado: %d", fatorial(n));
	return 0; 
}
