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
public interface Conta {
    void depositar(double valor);
    void sacar(double valor);
    double getSaldo();
}
