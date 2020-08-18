import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;
/**
 *PROFESSOR, tentei, de verdade, mas essa estrutura é muito complexa.
 * @author Breno
 */
public class Hash {
    public String item;
    public boolean ocupado;
    
    public Hash(boolean b) { // construtor
        ocupado = b;
    }
    public static void main(String[] args) {
        Scanner menu = new Scanner (System.in);
        while (true) {
            System.out.print("##--Digite a opcao desejada:--##\n\n");
            System.out.print("|--------------------------------------|\n");
            System.out.print("| Opção 1 - Busca de palavra reservada |\n");
            System.out.print("| Opção 2 - Leitura de TXT             |\n");
            System.out.print("| Opção 3 - Sair                       |\n");
            System.out.print("|--------------------------------------|\n");

            int opcao = menu.nextInt();

            if (opcao == 3) {
                System.out.println("\nAté logo!");
                menu.close();
            }

            switch (opcao) {
            case 1:
                System.out.println("Digite a palavra desejada: \n");
                break;
            case 2:
                System.out.println("Informe o arquivo de texto: \n");
                String texto = menu.nextLine();
                System.out.println("Conteudo do arquivo: \n");
                try (FileReader arq = new FileReader(texto)) {
                    BufferedReader lerArq = new BufferedReader(arq);
                    String linha = lerArq.readLine(); // lê a primeira linha
                    // a variável "linha" recebe o valor "null" quando o processo
                    // de repetição atingir o final do arquivo texto
                    while (linha != null) {
                        System.out.printf("%s\n", linha);
                        linha = lerArq.readLine(); // lê da segunda até a última linha
                    }
                } catch (IOException e) {
                    System.err.printf("Erro na abertura do arquivo: %s.\n",
                    e.getMessage());
                }
                System.out.println();
            default:
                System.out.println("Opção Inválida! \n");
                break;
        }
    }
 }
}
