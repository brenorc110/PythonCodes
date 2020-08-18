#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

int main(){
 int op, id;
 string dr, especialidade;
 cout<<"Bem-vindo ao Consulta facil. ";
 cout<<"Clique na opcao desejada: \n";
 cout<<"1 Pediatria 2 Clinico Geral 3 Geriatria 4 Consultar internacoes\n";
 cin>>op;
 //Após pedir pro usuário entrar com a especialidade médica, irá direcionar aos doutores que a atendem.
 switch(op){
  case 1:
  especialidade = "Pediatra";
  cout<<"Escolha o Dr.: \n ";
  cout<<"1 Dr. Paulo 2 Dr. Jose 3 Dra Andreia ou 4 para voltar ao menu anterior.\n";
  cin>>op;
  switch(op){
   case 1:
//Define  exibe o doutor e a especialidade escolhida. 
   dr = "Paulo";
   cout<<"Consulta " + especialidade + " Dr. "+ dr + ". \n";
   system("pause");
   case 2:
   dr = "Jose";
   cout<<"Consulta " + especialidade + " Dr. "+ dr + ". \n";
   system("pause");
   case 3:
   dr = "Andreia";
   cout<<"Consulta " + especialidade + " Dr. "+ dr + ". \n";
   system("pause");
   default:
   system("cls");
   main();
  }
  case 2:
  especialidade = "Clinico Geral";
  cout<<"Escolha o Dr.: \n ";
  cout<<"1 Dr. Lucas 2 Dr. Breno ou 3 para voltar ao menu anterior. \n";
  cin>>op;
  switch(op){
   case 1:
   dr = "Lucas";
   cout<<"Consulta " + especialidade + " Dr. "+ dr + ". \n";
   system("pause");
   case 2:
   dr = "Breno";
   cout<<"Consulta " + especialidade + " Dr. "+ dr + ". \n";
   system("pause");
   default:
   system("cls");
   main();
  }
  case 3:
  especialidade = "Geriatra";
  cout<<"Escolha o Dr.: \n ";
  cout<<"1 Dr. Marcelo ou 2 para voltar ao menu anterior. \n";
  cin>>op;
  switch(op){
   case 1:
   dr = "Marcelo";
   cout<<"Consulta " + especialidade + " Dr. "+ dr + ". \n";
   system("pause");
   default:
   system("cls");
   main();
  }
  case 4:
  especialidade = "internacao";
  dr = "";
  cout<<"Insira ID da internacao ou do paciente: \n";
  cin>>id;
  cout<<"Consultando "+ especialidade + " ID: ";
  cout<<id;
  cout<<"...";
  cout<<"\n";
  system("pause");
 }

//por fim, sempre vai limpar a tela e voltar ao menu inicial. 

    system("cls");
    main();
    return(0);
}
