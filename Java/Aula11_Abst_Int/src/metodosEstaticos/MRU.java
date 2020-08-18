/*
Programa MRU
fonte: https://www.infoescola.com/fisica/movimento-retilineo-uniforme/
 */
package metodosEstaticos;

import java.util.Scanner;

public class MRU {
    
public static float velMedia(float Si, float Sf, float Ti, float Tf){
    return (Sf-Si)/(Tf-Ti);
}
public static float variacaoEspaco(float Si, float Sf){
    return Sf-Si;
}
public static float varTempo(float Ti, float Tf){
    return Tf-Ti;
}

public static void main (String[] Args){
    int op, op2 = 1;
    float Si, Sf, Ti, Tf, v, t;
    Scanner e = new Scanner(System.in);
    System.out.println("Bem vindo ao programa de cálculo de MRU's. ");
    
    do{
    System.out.println("Para calculo de vel. media, digite 1 ");
    System.out.println("Para calculo de variacao de espaço, digite 2 ");
    System.out.println("Para calculo de variacao de tempo, digite 3: ");
    op = e.nextInt();
    
    switch (op){
        case 1:
    System.out.println("Para calculo da vel. media: ");
    System.out.println("Informe a posicao inicial: ");
    Si = e.nextFloat();
    System.out.println("Informe a posicao final: ");
    Sf = e.nextFloat();
    System.out.println("Informe a tempo inicial: ");
    Ti = e.nextFloat();
    System.out.println("Informe a tempo final: ");
    Tf = e.nextFloat();
    System.out.println("Vel. media= " + velMedia(Si, Sf, Ti, Tf) + " m/s.");
    break;
    
    case 2:
    System.out.println("Para calculo da posicao final (variacao de espaco): ");
    System.out.println("Informe a posicao inicial em metros: ");
    Si = e.nextFloat();
    System.out.println("Informe a posicao final em metros: ");
    Sf = e.nextFloat();
    System.out.println("Posicao inicial= " + variacaoEspaco(Si, Sf) + " metros.");
    break;
    
    case 3:
    System.out.println("Para calculo da variacao de tempo: ");
    System.out.println("Informe a tempo inicial: em segundos ");
    Ti = e.nextFloat();
    System.out.println("Informe a tempo final: em segundos");
    Tf = e.nextFloat();
    System.out.println("Variacao de tempo= " + varTempo(Ti, Tf) + " segundos");
    break;
    
    default:
    System.out.println("Opcao incorreta!");
    break;
    
    }
    System.out.println("Gostaria de fazer outra operacao? 1-SIM/2-NAO");
    op2 = e.nextInt();
    } while (op2==1);
}
}