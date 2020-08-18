
package Tela;

public class Financeira extends Banco{
   
    private int qtdParcelas;
    private double valorFin;
    private double valorParc;
    private String vencimento;
    private String vencprimParc;
    private int formaPG;
       
    public int getQtdParcelas() {
        return qtdParcelas;
    }

    public void setQtdParcelas(int qtdParcelas) {
        this.qtdParcelas = qtdParcelas;
    }

    public double getValorFin() {
        return valorFin;
    }

    public void setValorFin(double valorFin) {
        this.valorFin = valorFin;
    }

    public double getValorParc() {
        return valorParc;
    }

    public void setValorParc(double valorParc) {
        this.valorParc = valorParc;
    }

    public String getVencimento() {
        return vencimento;
    }

    public void setVencimento(String vencimento) {
        this.vencimento = vencimento;
    }

    public String getVencprimParc() {
        return vencprimParc;
    }

    public void setVencprimParc(String vencprimParc) {
        this.vencprimParc = vencprimParc;
    }

    public int getFormaPG() {
        return formaPG;
    }

    public void setFormaPG(int formaPG) {
        this.formaPG = formaPG;
    }
    
    
    
}
