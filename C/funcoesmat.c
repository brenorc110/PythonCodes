#include <math.h>
#include <stdlib.h>
#include <stdio.h>

int main(void){
double graus, raio, base, expoente, x;
printf("Trigonometria\nDigite o numero dos graus: ");
scanf("%lf", &graus);
printf("Geometria\nDigite a distancia do raio do circulo: ");
scanf("%lf", &raio);
printf("Potenciacao\nDigite a base: ");
scanf("%lf", &base);
printf("Digite o expoente: ");
scanf("%lf", &expoente);
x = pow(base,expoente);

printf("O seno de %.2lf eh %.2lf\n", graus,sin(graus));
printf("O cosseno de %.2lf e %.2lf\n", graus, cos(graus));
printf("A tangente de %.2lf eh %.2lf\n", graus,tan(graus));
printf("%.2lf elevado a %.2lf eh %.2lf\n" , base, expoente, x);
printf("A raiz quadrada de %.2lf eh %.2lf\n", x ,sqrt(x));
printf("O log2 de %.2lf eh %.2lf\n",x,log(x));
printf("O log10 de %.2lf eh %lf\n", x, log10(x));
printf("O teto de %.2lf eh %.2lf\n",x,ceil(x));
printf("A parte inteira de %.2lf eh %lf\n",x,floor(x));
printf("Um circulo com %.2lf cm de raio tem: \n", raio);
printf("%.2lf cm de comprimento e \n", raio*2*M_PI);
printf("%.2lf cm de area.", pow(raio,2)*M_PI);

//system ("pause");
return EXIT_SUCCESS;
}

/*
#include <cmath> // ou <math.h>
#include <iostream>
#include <cstdlib>
using namespace std;

int main (void){
    double graus, raio, base, expoente, x;
    cout << "Trigonometria\nDigite o numero dos graus -> ";
    cin >> graus;
    cout << "Geometria\nDigite a distancia do raio do circulo -> ";
    cin >> raio;
    cout << "Potenciacao\nDigite a base -> ";
    cin >> base;
    cout << "Digite o expoente -> ";
    cin >> expoente;
    cin.ignore();
    system ("cls");
    x = pow (base, expoente);
    cout << "O seno de "<<graus<<" e "<<sin(graus)<<endl;
    cout << "O co-seno de "<<graus<<" e "<<cos(graus)<<endl;
    cout << "A tangente de "<<graus<<" e "<<tan(graus)<<endl;
    cout << base<<" elevado a "<<expoente<<" e "<<x<<endl;
    cout << "A raiz quadrada de "<<x<<" e "<<sqrt(x)<<endl;
    cout << "O log2 de "<<x<<" e "<<log(x)<<endl;
    cout << "O log10 de "<<x<<" e "<<log10(x)<<endl;
    cout << "O teto de "<<x<<" e "<<ceil(x)<<endl;
    cout << "A parte inteira de "<<x<<" e "<<floor(x)<<endl;
    cout << "Um circulo com "<<raio<<"cm de raio tem: ";
    cout << raio*2*M_PI<<"cm de comprimento e ";
    cout << pow(raio,2)*M_PI<<"cm de area.\n"<<endl;
    system ("pause");
    return EXIT_SUCCESS;
}*/

