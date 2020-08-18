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
public class PessoaJuridica extends Pessoa{
    private String CNPJ;

    public String getCNPJ() {
        return CNPJ;
    }

    public void setCNPJ(String CNPJ) {
        this.CNPJ = CNPJ;
    }
    
    public static void main(String[] args) {
        PessoaJuridica pj1 = new PessoaJuridica();
        pj1.setNome("Breno");
        pj1.setEndereco("Rua 31");
        pj1.setCidade("SJC");
        pj1.setCEP("10000-710");
        pj1.setEstado("SP");
        pj1.setTelefone("(12) 3000-0000");
        pj1.setCNPJ("400000000-99");
                
    }
}
