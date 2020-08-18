/*
    Conte os paragrafos de um texto: (separador \n) 
 */
package aula13;
/**
 *
 * @author fsj.5947373464
 */
public class Ex4 {
    public static void main(String[] args) {
         String entrada;
        int a, paragrafos = 0;
        entrada ="teste \n teste \n";
        
        for (int i=0; i<entrada.length(); i++){
            
            if(entrada.charAt(i) == '\n'){
            paragrafos++;
            }
        }
        System.out.println("paragrafos= "+ paragrafos); 
        
    }
    
}
