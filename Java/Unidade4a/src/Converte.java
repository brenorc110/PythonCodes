import java.util.StringTokenizer;
import java.util.ArrayList;
import java.util.Scanner;

public class Converte {
    // metodo lexico - separa a entrada em tokens
    private static ArrayList<String> lexico(String sentenca){     
        //cria uma a FILA vazia para armazenar os tokens separadamente
        ArrayList<String> infixa = new ArrayList<String>();
        // separa a entrada em tokens
        // IMPORTANTE A ENTRADA DEVE SEPARAR OS TOKENS COM ESPACOS
        // EXEMPLO A + B  não é aceito A+B
        StringTokenizer lista = new StringTokenizer(sentenca);                
        while ( lista.hasMoreTokens())
                infixa.add(lista.nextToken()); 
        return infixa;
    }
   
    //metodo prioridade - define a prioridade dos operadores
    // aqui é um exemplo de aplicacao de uma acao semantica        
    private static int prioridade(String operador){
          int pri=0;
         switch (operador){                
                case "+":                
                case "-":                
                    pri=2;
                    break;
                case "*":            
                case "/":
                    pri=3;
                    break;
                case "(":
                    pri=1;
         }
         return pri;
    }
    
    //metodo posFixa - é nosso algoritmo de conversão
    public static String posFixa(String sentenca){
        // cria saida vazia        
        String posFixa = "";
        // cria pilha vazia
        Pilha pilha = new Pilha();
        // chama metodo lexico - que separa a sentenca em tokens
        // IMPORTANTE ; somente é aceita entrada em espaco entre os tokens
        // exemplo: A + B é aceito, MAS A+B não é convertido
        ArrayList<String> infixa = lexico(sentenca);
        int n = infixa.size();  //  n armazena o numero de tokens a ser analisado
        // faz a analise de todos os tokens e monta a saida na notacao posFixa
        for (int i = 0 ; i< n; i++){                      
            // analisa o token
            switch (infixa.get(i)){                                                
                // 2.b - se é um OPERADOR
                case "+":                
                case "-":                
                case "*":            
                case "/":
                    /* 2.b.i.	Enquanto a pilha não estiver vazia e no topo 
                       existir um operador com prioridade <= ao encontrado, 
                       desempilha o operador e coloca na saída;
                    */
                    while (prioridade(pilha.topoGet()) >= prioridade(infixa.get(i))){
                        posFixa = posFixa + " " + pilha.desempilha();                        
                    }                    
                    // empilha o operador atual para continuar o processo                    
                    pilha.empilha(infixa.get(i));                    
                    break;                                        
                    
                case "(":
                    //2.C - encontrou abre-parentese '(' --> empilha o token encontrado
                    pilha.empilha(infixa.get(i));                    
                    break;
                    
                case ")":
                    /* 2.d - encontrou o fecha-parentese '(' 
                       é hora de desempilhar e adicionar o operador a saida
                       até encontrar o parentese correspondente ao abre_parentese                    
                    */
                    while ( !pilha.topoGet().equals("(")){
                        posFixa = posFixa + " " + pilha.desempilha();
                    } 
                    // desempilha o abre_parentese
                    pilha.desempilha();
                    break;                   
                default:
                    //numero ou variavel - é adicionado diretamente a saida
                    posFixa = posFixa  + " " + infixa.get(i);                
            }
        }    
        // 3. - esvazia a pilha e adiciona a saida
        while(!pilha.vazio()){            
            posFixa = posFixa + " " + pilha.desempilha();
        }
        return posFixa;
    }
    
    //metodo para testar
    public static void main(String[] args){
        // leitura da expressao infixa a ser lida
        String exprIn;
        Scanner in = new Scanner(System.in);  
        System.out.println("Digite uma expressao infixa - ATENCAO cada simbolo deve estar separados por espacos entre eles :");
        exprIn = in.nextLine();                    
        // converte a expressao         
        System.out.println(Converte.posFixa(exprIn));
        
    }        
}
