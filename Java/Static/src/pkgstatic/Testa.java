/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package pkgstatic;

/**
 *
 * @author Casa
 */
public class Testa {

    public static void main(String[] args) {
        System.out.println("Contador: " + Conta.contador);
        Conta c1 = new Conta();
        System.out.println("Número da primeira conta: " + c1.numero);

        System.out.println("Contador: " + Conta.contador);
        Conta c2 = new Conta();
        System.out.println("Número da segunda conta: " + c2.numero);

        System.out.println("Contador: " + Conta.contador);

        Conta.zeraContador();
        System.out.println("Contador zerado: " + Conta.contador);
    }
}
