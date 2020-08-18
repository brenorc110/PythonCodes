/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Casa
 */
public class seguroVeicular extends Servico {

    private Veiculo veiculo;
    private double valorDoSeguroVeicular;
    private double franquia;

    @Override
    public double calculaTaxa() {
        //return valorDoSeguroVeicular * franquia;
        return 5 + this.veiculo.getTaxa() * 0.05;
    }
}
