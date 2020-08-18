/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package pkgstatic;

/**
 *
 * @author Casa
 */
public class Conta {

    static int contador;
    int numero;

    Conta() {
        Conta.contador++;
        this.numero = Conta.contador;
    }

    static void zeraContador() {
        System.out.println("Contador: " + Conta.contador);
        System.out.println("Zerando o contador....");
        Conta.contador = 0;
    }
}
