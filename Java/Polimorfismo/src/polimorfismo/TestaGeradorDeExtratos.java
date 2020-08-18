/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Casa
 */
public class TestaGeradorDeExtratos {

    public static void main(String[] args) {
        GeradorDeExtrato gerador = new GeradorDeExtrato();

        ContaPoupanca cp = new ContaPoupanca();
        cp.setSaldo(1000);
        ContaCorrente cc = new ContaCorrente();
        cc.setSaldo(1000);


        gerador.imprimeExtratoBasico(cp);
        gerador.imprimeExtratoBasico(cc);

    }
}
