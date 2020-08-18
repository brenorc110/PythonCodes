/*
Programa MRUV
fonte: https://www.infoescola.com/fisica/movimento-retilineo-uniformemente-variado/
https://profes.com.br/tira-duvidas/fisica/gostaria-de-todas-as-formulas-de-mru-e-mruv/
 */
package metodosEstaticos;

import java.util.Scanner;

public class MRUV {
public static float aceleracao(float dist,float  tempo){
 return dist/tempo;
}
public static float velAdiquirida(float v0,float  a,float  tempo){
    return v0+(a*tempo);
}
public static float varEspaco(float s0, float v0, float a, float tempo){
    return s0+(v0*tempo)+a*(tempo*tempo)/2;
}
public static float velTorricelli(float v0, float a, float dist){
    float v2 = (v0*v0)+2*a*dist;
    return (float) Math.sqrt(v2);
}
    
    public static void main (String[] Args){
    int op, op2 = 1;
    float dist, tempo, v0, a, s0;
    Scanner e = new Scanner(System.in);
    System.out.println("Bem vindo ao programa de cálculo de MRUV. ");
    do{
    System.out.println("Para cálculo de aceleracao, digite 1");
    System.out.println("Para cálculo da veloc. adquirida, digite 2");
    System.out.println("Para calculo da variacao de espaco, digite 3");
    op = e.nextInt();
    
    switch (op){
        case 1:
    System.out.println("Informe a distancia percorrida em metros: ");
    dist = e.nextFloat();
    System.out.println("Informe o tempo do percurso em segundos: ");
    tempo = e.nextFloat();
    System.out.println("Aceleracao = " + aceleracao(dist, tempo) + " m/s ao quadrado. ");
    break;
    
        case 2:
    System.out.println("Informe a velocidade inicial: ");
    v0 = e.nextFloat();
    System.out.println("Informe a aceleracao: ");
    a = e.nextFloat();
    System.out.println("Informe o tempo em segundos: ");
    tempo = e.nextFloat();
    System.out.println("Valocidade adquirida = " + velAdiquirida(v0, a, tempo) + " m/s. ");
    break;
    
        case 3:
    System.out.println("Calculo da variacao de espaco: Formula: S=So+Vot+at2/2");
    System.out.println("Informe a distancia inicial da origem em metros: ");
    s0 = e.nextFloat();
    System.out.println("Informe a velocidade inicial: ");
    v0 = e.nextFloat();
    System.out.println("Informe o tempo em segundos: ");
    tempo = e.nextFloat();
    System.out.println("Informe a aceleracao: ");
    a = e.nextFloat();
    System.out.println("Variacao de espaco = " + varEspaco(s0, v0, a, tempo) + " metros.");
    break;
    
    case 4:
    System.out.println("Velocidade adquirida - com equacao de Torricelli: ");
    System.out.println("Informe a velocidade inicial: ");
    v0 = e.nextFloat();
    System.out.println("Informe a aceleracao: ");
    a = e.nextFloat();
    System.out.println("Informe a distancia percorrida em metros: ");
    dist = e.nextFloat();
    System.out.println("Velocidade adquirida = " + velTorricelli(v0, a, dist) + " m/s. ");
    break;
    
    default:
        System.out.println("Opcao incorreta!");
    break;
    }
    
    System.out.println("Gostaria de fazer outra operacao? 1-SIM/2-NAO");
    op2 = e.nextInt();
    }while (op2==1);
}
}