/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Casa
 */
public class conta {

    double saldo;
    double limite = 100;
    int numero;
    Agencia agencia;

    void Conta(int numero) {
        this.numero = numero;
    }

    void Conta(int numero, double limite) {
        //this(numero); <--comentei só pra não mostrar o erro.
        this.limite = limite;
    }

    void Conta(Agencia agencia) {
        this.agencia = agencia;
    }

    void deposita(double valor) {
        //implementação
        this.saldo += valor;
        this.descontaTarifa();
    }

    void saca(double valor) {
        this.saldo -= valor;
    }

    void imprimeExtrato() {
        this.imprimeExtrato(15); //<-- Retorna um extrato dos últimos 15 dias.
        // System.out.println("SALDO= "+ this.saldo);
    }

    void imprimeExtrato(int dias) {
    }

    double consultaSaldoDisponivel() {
        return this.saldo + this.limite;
    }

    void transfere(conta destino, double valor) {
        this.saldo -= valor;
        destino.saldo += valor;
    }

    private void descontaTarifa() {
        this.saldo -= 0.1;
    }
}
