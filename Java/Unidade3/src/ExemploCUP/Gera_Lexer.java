package ExemploCUP;
import java.io.*;
import java.nio.file.Paths;
import java_cup.*;
import java_cup.runtime.*;
public class Gera_Lexer {
     public static void main(String[] args) throws IOException {  
        String arq = Paths.get("").toAbsolutePath().toString() + "/src/ExemploCUP/";
        String arq1 = arq  + "lexico.flex";  // nome do arq. do lexico
        System.out.println(arq1);
        gerarLexer(arq1);               
     }
     public static void gerarLexer( String arq){
         File file = new File(arq);
         jflex.Main.generate(file);         
     }
      
}
