/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Casa
 */
public class Servico {

    private Cliente contratante;
    private Funcionario responsavel;
    private String dataDeContrato;

    public double calculaTaxa() {
        return 10;
    }

    /**
     * @return the dataDeContrato
     */
    public String getDataDeContrato() {
        return dataDeContrato;
    }

    /**
     * @param dataDeContrato the dataDeContrato to set
     */
    public void setDataDeContrato(String dataDeContrato) {
        this.dataDeContrato = dataDeContrato;
    }
}
