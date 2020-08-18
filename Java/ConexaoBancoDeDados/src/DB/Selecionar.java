package DB;



/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author fsj.8500494981
 */
public class Selecionar extends ConexaoBancoDeDados implements Runnable {

    
     Selecionar(){
       super();
       System.out.println("Init Thread");
     
   
    }
   
   
    @Override
    public void run() {
        this.prepareSelect();
    }
    
    
       public static void main(String[] args) {
        Inserir tre  = new Inserir();
        Thread t2 = new Thread();
        t2.start();
        tre.prepareSelect();
        
    }
    
    
    
}
