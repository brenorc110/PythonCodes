package thread1;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Marcos
 */
public class ThreadA extends Thread {

   public ThreadA(){ //método construtor
   }

   public void run(){ //método RUN -> toda lógica de execução da Thread
       for (int i=1; i<=1000000; i++){
          System.out.println(i);
       }
   }

   public static void main (String args[]){ //método principal

       ThreadA a = new ThreadA(); //criando novo objeto Thread

       //iniciando a Thread
       System.out.println("Thread A: ");
       a.start();
       
       ThreadB b = new ThreadB(); //criando novo objeto Thread
       System.out.println("Thread B: ");
       b.start();
       
       ThreadC c = new ThreadC();
       System.out.println("Thread C: ");
       c.start();
    }
}

