package DB;


import Tela.Pessoa;
import java.sql.Statement;
import java.sql.Connection;
import java.util.Scanner;
/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author fsj.8500494981
 */
public class Inserir extends ConexaoBancoDeDados implements Runnable {

    Inserir(){
     super();
     System.out.println("Init Thread");
     
    }
   

    @Override
    public void run() {
      // this.inserir();
    }
    
    
       public static void main(String[] args) {
        Inserir tre  = new Inserir();
        Thread t1 = new Thread();
        t1.start();
        
    }
    
    
}
