/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Casa
 */
public class Emprestimo extends Servico {

    //private Veiculo veiculo;
    private double valorDoSeguroVeicular;
    private double franquia;
    private double taxa;
    private double Valor;

    /**
     * @return the Valor
     */
    public double getValor() {
        return Valor;
    }

    /**
     * @param Valor the Valor to set
     */
    public void setValor(double Valor) {
        this.Valor = Valor;
    }

    public double calculaTaxaDeEmprestimo(double Valor) {
        return this.Valor = Valor * 0.1;
    }

    public String setDataDecontratacao() {
        return "10/10/2013";
    }

    ;
public double calculaTaxa() {
        //return 5 + this.calculaTaxaDeEmprestimo(10);
        return super.calculaTaxa() * this.Valor * 0.1;
    }
;
}
