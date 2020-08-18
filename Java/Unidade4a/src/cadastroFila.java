import static java.lang.System.in;
import java.util.Scanner;

public class cadastroFila{
    public static void main(String[] args) {
        Fila f = new Fila();
        int op=1;
        while(op==1){
            System.out.println("Gostaria de adicionar ou remover tarefa? 1-ADD 2-Fazer tarefa");
            Scanner in  = new Scanner(System.in);
            op = in.nextInt();
            if (op==1) {
                System.out.println("Digite a tarefa: ");
                String tarefa = in.next();
                
                f.adicionar(tarefa);
            }else{
                System.out.println("Tarefa a realizar");
                f.mostrar();
                f.remover();
            }
            System.out.println("Deseja add outra? 1-SIM 2- SAIR");
            op = in.nextInt();
        }
    }
}

class Fila{
    int inicio;
    int fim;
    int tamanho;
    int qtdeElementos;
    String f[];

    public Fila(){
        inicio = fim = -1;
        tamanho = 100;
        f = new String[tamanho];
        qtdeElementos = 0;
    }

    public boolean estaVazia(){
        if (qtdeElementos == 0){
            return true;
        }
        return false;
    }

    public boolean estaCheia(){
        if (qtdeElementos == tamanho - 1){
            return true;
        }
        return false;
    }

    public void adicionar(String tarefa) {
        if (!estaCheia()) {
            if (inicio == -1) {
                inicio = 0;
            }
            fim++;
            f[fim] = tarefa;
            qtdeElementos++;
        }
    }

    public void remover(){
        if (! estaVazia() ){
            inicio++;
            qtdeElementos--;
        }
    }

    public void mostrar(){
        String elementos = "";
            for (int i = inicio; i<=fim; i++) {
                elementos += f[i]+ " - ";
            }
            System.out.println(elementos);
    }
}
