#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

int main(){
	unsigned long int cartao;
	int opcao;
	float valor, saldo = 1000;
	
	cout<<"Bem vindo ao nosso banco.\n";
	cout<<"Insira o numero de seu cartao: \n";
	cin>>cartao;
	
	if(cartao<40000000){
		cout<<"Cartao invalido, tente novamente!";
		system("cls");
		main();
	} else {
		cout<<"Digite a opcao desejada: \n";
		cout<<"1 para saque 2 para deposito 3 para visualizar saldo \n";
		cin>>opcao;
		switch(opcao){
			case 1:
				cout<<"Digite o valor a ser sacado: Ex; 2000: \n";
				cin>>valor;
				if(saldo-valor <0){
					cout<<"Operacao invalida, tente novamente!";
					system("pause");
		system("cls");
		main();
				} else {
					saldo = saldo-valor;
					cout<<"OK, seu saldo eh \n";
					cout<<saldo;
					cout<<"\n";
					opcao=1;
				}
				
				case 2:
				cout<<"Digite o valor a ser depositado: Ex; 2000";
				cin>>valor;
				saldo = saldo+valor;
				cout<<"OK, seu saldo eh \n";
					cout<<saldo;
					cout<<"\n";
			    case 3:
					cout<<"OK, seu saldo eh \n";
					cout<<saldo;
					cout<<"\n";
		}
	}
	system("pause");
	system("cls");
	main();
	return(0);
}
