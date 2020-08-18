/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package aula11_abst_int;

/**
 *
 * @author Breno
 */
public class ContaCorrente implements Conta {
    private double Saldo;
    private double taxaOperacao = 0.45;
    
    @Override
    public void depositar(double valor){
    this.Saldo +=valor - taxaOperacao;
    }
    @Override
    public double getSaldo(){
    return Saldo;
    }
    @Override
    public void sacar(double valor){
    this.Saldo -=valor + taxaOperacao;
    }
    
}
