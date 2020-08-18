/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Casa
 */
public class TestaMetodoTransfere {
    public static void main (String[] args){
            Agencia a = new Agencia(1234);
    conta origem = new conta();
    origem.saldo = 1000;
    conta destino = new conta();
    origem.transfere(destino, 500);
    }
    }

