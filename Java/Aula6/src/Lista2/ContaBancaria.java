/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Lista2;

/**
 *
 * @author fsj.5947373464
 */
public class ContaBancaria {
    String nomeCliente;
    int senhaNum;
    String senhaL;
    float saldo = 10000, valor, chequeEspecial = 2000;
    
    float verSaldo(){
        
        return saldo;
    }
    
    float verValorChequeEspecial(){
        
        return chequeEspecial;
    }
    
    float verQtdeChequeEspecialUtilizado(){
        
        return chequeEspecial;
    }
    
    float Debito(float valor){
        if(valor>chequeEspecial || valor>1500f){
            System.out.println("Saque impossível!!! ");
            return saldo;
        }
        else {
            System.out.println("Echo! ");
            return saldo-valor;
        }
    }
    
    float Deposito(float valor){
        
        return saldo+valor;
    }
    
    public static void main(String[] args) {
       ContaBancaria c1 = new ContaBancaria();
       c1.Debito(10f);
        System.out.println("Saldo: "+ c1.saldo);
    }
}
