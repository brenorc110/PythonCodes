package thread1;
import java.util.*;

public class Cronometro {
    public Timer timer;
    //Dados em milisegundos:
    private int primeiraVez=0;
    private int intervalo=0;

    public Cronometro(int primeiraVez, int intervalo) {
        this.primeiraVez=primeiraVez*1000;
        this.intervalo=intervalo*1000;
        this.timer=new Timer();
        this.timer.scheduleAtFixedRate(new Tarefa(), this.primeiraVez, this.intervalo);
        System.out.println("tarefa agendada");
    }
    private class Tarefa extends TimerTask{
        public void run(){
            System.out.println("Execução da tarefa agendada");
        }
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Cronometro c = new Cronometro(5,1);
        int i=0;
        while(i!=1){
            System.out.println(" Digite 1 para parar a tarefa.");
            Scanner scan = new Scanner(System.in);
            i=scan.nextInt();
            
            if(i==1)
                c.timer.cancel();
        }
        System.out.println("Tarefa cancelada");
    }
    
}
