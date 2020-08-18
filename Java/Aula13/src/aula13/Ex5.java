package aula13;
/*
 Conte quantas palavras um texto possui.
 */
/**
 *
 * @author fsj.5947373464
 */
public class Ex5 {
    public static void main(String[] args) {
         String entrada;
        int i, palavras = 1;
        entrada = "breno renan";
        //entrada = entrada.replace( " ", "," );
        System.out.println(entrada);
        
        for(i=0; i<entrada.length(); i++) {
          char b = entrada.charAt(i);
          if (b == ' '){
              palavras++;
          }
          }
          System.out.println("palavras= "+ palavras);
          }
      }
      