/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Casa
 */
public class TestaContaEAgencia {
 public static void main(String[] args){
     //Criando alguns objetos
     Agencia a = new Agencia(1234);
     conta c  = new conta();
     
     //carregando alguns dados
     a.numero = 178;
     c.saldo = 1000.0;
     
     //Ligando os objetos
     c.agencia = a;
     //System.out.println(c.agencia.numero);
     //System.out.println(c.saldo);
 }   
}
