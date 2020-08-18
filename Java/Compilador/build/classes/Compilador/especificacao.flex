package Compilador;
import static Compilador.Token.*;
%%
%{
    private void imprimir(String token, String lexema){
                    System.out.println(lexema + " ==> " + token);
    }
%}
%class Lexer
%type Token
nomeVariavel = [_a-zA-Z][_a-zA-Z0-9]*
inteiro = [0-9]+
decimal = [0-9]+["."]+[0-9]+
blocoComentario = "/*" ~"*/"
branco = [\t\n\r ]+
linhaComentario = {branco}* "//" .*
palavraChave = "if" | "class" | "int" | "while" | "do"
%%

{palavraChave}              { imprimir("PALAVRA CHAVE :  ", yytext()); return PALAVRA_CHAVE; }
{nomeVariavel}              { imprimir("VARIAVEL :  ", yytext()); return NOME_VARIAVEL; }
{inteiro}                   { imprimir("NUMERO INTEIRO :  ", yytext()); return INT; }
{decimal}                   { imprimir("NUMERO DECIMAL :  ", yytext()); return DEC; }
{blocoComentario}           { imprimir("COMENTARIO :  ", yytext()); return COMENTARIO; }
{linhaComentario}           { imprimir("COMENTARIO :  ", yytext()); return COMENTARIO; }
{branco}                    { return BRANCO; }
.           { imprimir ("<<< CARACTER INVALIDO! >>>  ", yytext()); return   ERROR; }
<<EOF>>     { return null; }