package aula13;

import java.util.Scanner;
public class manipulaString{

     @SuppressWarnings("ResultOfMethodCallIgnored")
     public static void main(String []args){
         Scanner entrada = new Scanner(System.in);
        System.out.println("Informe o nome do artista/ cantor: ");
        String artista = entrada.next();
        
        System.out.println("Informe o nome da musica: ");
        String musica = entrada.next();
        
        System.out.println("Informe o genero da musica: ");
        String genero = entrada.next();
        
        //Se primeira letra do nome do artista é não maiuscula, a transforma em mauiscula. 
        char cp = artista.charAt(0);
        if (Character.isLowerCase(cp)){
            artista = artista.substring(0, 1).toUpperCase() + artista.substring(1);
        }
        //Se o genero tiver duas palavras, será retornada apenas a primeira. 
            
        for (int i=0; i>genero.length(); i++){
         do{
            genero.substring(0, genero.indexOf(""));
                
            } while (genero.charAt(i) != ' ');
        }
        System.out.println("Cantor:" + artista);
        System.out.println("Nome da musica:" + musica);
        System.out.println("Genero da musica:" + genero);
        
        }
}