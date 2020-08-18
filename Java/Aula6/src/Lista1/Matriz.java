package Lista1;
import java.util.Scanner;
/**
 *
 * @author fsj.5947373464
 */
public class Matriz {
    public static void main(String[] args) {
        int matriz[][] = new int[5][5];
        int linha, coluna, i, j;
        Scanner entrada = new Scanner(System.in);
        System.out.println("Bem-vindo,o numero:");
                
        for (linha = 0; linha < 5; linha++) {
            for (coluna = 0; coluna < 5; coluna++) {
                System.out.printf("Entre com o valor matriz [%d][%d]: ",
                        linha + 1, coluna + 1);
                matriz[linha][coluna] = entrada.nextInt();
            }
        }
        
        for (i = 0; i < 5; i++) {
            for (j = 0; j < 5; j++) {
                System.out.print(matriz[i][j]);
            }
			System.out.println("");
        }
        
    }
}
