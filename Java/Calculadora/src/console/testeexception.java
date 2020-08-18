package console;
import java.util.InputMismatchException;
import java.util.Scanner;
public class testeexception
{
    public static void main(final String args[]) {
        boolean continualaço = true;
        Scanner entrada = new Scanner(System.in);
        do {
            try {
                System.out.print("Informe o código do produto: ");
                entrada.nextInt();
                continualaço = false; // entrada bem-sucedida implica em fim da leitura.
            } catch (final InputMismatchException erroEntradaIncorreta) {
                entrada.nextLine(); // ignora entrada anterior.
                System.out.println("Você deve informar número inteiro apenas.\n");
            }
        } while (continualaço);
        System.out.println("OK! Tudo correto com seu numero! ");
        entrada.close();
    }
}