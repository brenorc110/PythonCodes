package Compilador;
import java.io.*; // iremos precisajflex.Main.generate(file);r desta classe para leitura do arquivo que contem a especificação

public class GeraLexer {
	public static void main(String[] args) throws IOException  {
		// IMPORTANTISSIMO : aqui você precisa identificar o caminho do seu projeto
		// no caso deste exemplo esta na pasta C:/Compiladores
		// por ex.substitua C:/Compiladores
		// por unidade:/ nome da pasta do seu projeto/ o restante mantenha
		String arq = "C:\\Users\\Breno\\OneDrive\\NetBeansProjects\\Compilador\\src\\Compilador\\especificacao.flex";
		gerarLexer(arq);
	}
public static void gerarLexer( String arq) {
            // instancia um objeto File para ler a especificação da gramatica
            File file = new File(arq);
            jflex.Main.generate(file);
            
            /* chama o método generate da classe Main do Jflex para criar o
            analisador léxico de acordo com as regras especificadas no
            arquivo especificao.flex criado por você.
	 */
	}
}