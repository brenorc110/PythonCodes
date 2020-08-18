/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package aula8heranca;

/**
 *
 * @author fsj.5947373464
 */
public class PessoaFisica extends Pessoa{
    private String CPF;

    public String getCPF() {
        return CPF;
    }

    public void setCPF(String CPF) {
        this.CPF = CPF;
    }
    
    public static void main(String[] args) {
        PessoaFisica pf1 = new PessoaFisica();
        pf1.setNome("Breno");
        pf1.setEndereco("Rua 31");
        pf1.setCidade("SJC");
        pf1.setCEP("10000-710");
        pf1.setEstado("SP");
        pf1.setTelefone("(12) 3000-0000");
        pf1.setCPF("400000000-99");
        
    }
    
}
