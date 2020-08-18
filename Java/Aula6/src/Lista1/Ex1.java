package Lista1;
import java.util.Scanner;
/**
 *
 * @author fsj.5947373464
 */
public class Ex1 {

    public static void main(String[] args) {
       Scanner entrada = new Scanner(System.in);
       int i =1;
           float soma = 0;
       do{
        System.out.println("Insira seu pedido: \n");
        System.out.println("1 - Coca lata 350ml \n");
        System.out.println("2 - Coca garrafa 600ml \n");
        System.out.println("2 - Coca garrafa 2L \n");
        
        int ped = entrada.nextInt();
    
           switch (ped) {
               case 1:
                   soma+= 0.350;
                   
                   break;
               case 2:
                   soma+= 0.600;
                   break;
               default:
                   soma+=2.0;
                   break;
           }
        System.out.println("Deseja pedir outro produto? S/N");
        String op = entrada.next();
        if ("S".equals(op)) i++; else i=0;
    } while(i>0);
    System.out.println("Fechando pedido... \n");
    System.out.println("Você pediu " + soma + " litros de refri!");
}
}
