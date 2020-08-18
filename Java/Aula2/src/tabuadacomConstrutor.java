import java.util.Scanner;
public class tabuadacomConstrutor {
    public int i=1, x, res;
    @SuppressWarnings("SillyAssignment")
    tabuadacomConstrutor(){
        this.i = i;
        this.x = x;
        this.res = res;
    }
    public static void main(String[] args) {
        Scanner entrada = new Scanner(System.in);
        tabuadacomConstrutor t = new tabuadacomConstrutor();
        
        System.out.println("Escreva o número para obter a tabuada: ");
        t.x = entrada.nextInt();
        do{
            t.res = t.x*t.i;
            t.i++;
            System.out.println(t.x + "x" + (t.i-1) + " = " + t.res);
           } while (t.i<=10);
    }
}