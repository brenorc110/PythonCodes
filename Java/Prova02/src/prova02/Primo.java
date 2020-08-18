/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package prova02;

public class Primo implements Runnable {
    public static int PRIMEIRO = -1;
    private final long tamanho;
    private int idThread = 0;
    public static synchronized void MarcarTermino(int idThread) {
        if (Primo.PRIMEIRO == -1) {
            Primo.PRIMEIRO = idThread;
        } else {
            System.out.println("Terminou primeiro: " + Primo.PRIMEIRO);
        }
    }

    Primo(long tamanho, int id) {
        this.tamanho = tamanho;
        this.idThread = id;
    }

    public boolean éPrimo(long primo) {
        boolean isPrimo = true;
        int divisor = 0;
        for (int i = 2; i <= primo; i++) {
            if (((primo % i) == 0) && (i != primo)) {
                isPrimo = false;
                divisor = i;
                break;
            }
        }
        return isPrimo;

    }

    public void run() {
        long fim = this.tamanho;
        long inicio = 0;
        while (inicio != fim) {
            if (éPrimo(inicio)) {
                System.out.println(inicio);
            }
            inicio++;
        }
        Primo.MarcarTermino(this.idThread);
    }

    public static void main(String[] args) {
        long tamanho = (long) Math.pow(10, 3);
        Primo p = new Primo(tamanho , 1);
        Primo p2 = new Primo(tamanho, 2);
        Thread t = new Thread(p);
        Thread t2 = new Thread(p2);
        t.start();
        t2.start();
    }

}
