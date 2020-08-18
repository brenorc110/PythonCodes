int idade;
void eleitor (int idade){
if (idade < 16 ) printf ("Nao eleitor.");
else if (idade>=18 && idade<=65) printf ("Eleitor obrigatorio."); else printf ("Eleitor facultativo.");
}
void main(){
printf ("Informe sua idade: ");
scanf ("%d", &idade);

eleitor(idade);
printf("\n");
}
