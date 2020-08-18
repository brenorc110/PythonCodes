package thread1;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Marcos
 */
public class ThreadB extends Thread {

   public ThreadB(){ //método construtor
   }

   public void run(){ //método RUN -> toda lógica de execução da Thread
       for (int i=1; i<=1000000; i++){
          System.out.println(i);
       }
   }

   public static void main (String args[]){ //método principal

       ThreadB b = new ThreadB(); //criando novo objeto Thread

       //iniciando a Thread
       System.out.println("Thread B: ");
       b.start();
    }
}
