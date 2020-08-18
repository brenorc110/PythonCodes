
package Tela;

public class Pessoa{
    
    private String nome;
    private double cpf;
    private rg rg;

    public rg getRg() {
        return rg;
    }

    public void setRg(rg rg) {
        this.rg = rg;
    }

    public Endereco getEndereco() {
        return endereco;
    }

    public void setEndereco(Endereco endereco) {
        this.endereco = endereco;
    }
    private Endereco endereco;
    private String dataNasc;
    private String telCel;
    private String telRes;
    private double patrimonio;
    private String tempoRes;
    private int tipoRes;
    private String nomePai;
    private String nomeMae;
    private String nacionalidade;
    private String dataEmissao;
    private String orgaoEmissor;
    private String nt;
    private String uf;
    private String pais_doc;
    private String fone_res;
    private String Fone_cel;

    public String getFone_cel() {
        return Fone_cel;
    }

    public void setFone_res(String fone_res) {
        this.fone_res = fone_res;
    }

    public String getFone_res() {
        return this.fone_res;
    }
    
    public void setPais_doc(String pais_doc) {
        this.pais_doc = pais_doc;
    }

    public String getPais_doc() {
        return this.pais_doc;
    }
    
    public void setUf(String uf) {
        this.uf = uf;
    }

    public String getUf() {
        return this.uf;
    }
    
    public void setNt(String nt) {
        this.nt = nt;
    }

    public String getNt() {
        return this.nt;
    }
    
    public void setOrgaoEmissor(String orgaoEmissor) {
        this.orgaoEmissor = orgaoEmissor;
    }

    public String getOrgaoEmissor() {
        return this.orgaoEmissor;
    }
    
    public void setDataEmissao(String dataEmissao) {
        this.dataEmissao = dataEmissao;
    }

    public String getDataEmissao() {
        return this.dataEmissao;
    }
    
    public String getNacionalidade(){
        return this.nacionalidade;
    }
    public void setNacionalidade(String nacionalidade) {
        this.nacionalidade = nacionalidade;
    }
       
    public Pessoa(String nome, double cpf) {
        this.nome = nome;
        this.cpf = cpf;
    }

    public String getNome() {
        return nome;
    }

    public void setNome(String nome) {
        this.nome = nome;
    }

    public double getCpf() {
        return cpf;
    }

    public void setCpf(double cpf) {
        this.cpf = cpf;
    }

    public String getDataNasc() {
        return dataNasc;
    }

    public void setDataNasc(String dataNasc) {
        this.dataNasc = dataNasc;
    }

    public String getTelCel() {
        return telCel;
    }

    public void setTelCel(String telCel) {
        this.telCel = telCel;
    }

    public String getTelRes() {
        return telRes;
    }

    public void setTelRes(String telRes) {
        this.telRes = telRes;
    }

    public double getPatrimonio() {
        return patrimonio;
    }

    public void setPatrimonio(double patrimonio) {
        this.patrimonio = patrimonio;
    }

    public int getTipoRes() {
        return tipoRes;
    }

    public void setTipoRes(int tipoRes) {
        this.tipoRes = tipoRes;
    }

    /**
     * @return the nomePai
     */
    public String getNomePai() {
        return this.nomePai;
    }

    /**
     * @param nomePai the nomePai to set
     */
    public void setNomePai(String nomePai) {
        this.nomePai = nomePai;
    }

    /**
     * @return the nomeMae
     */
    public String getNomeMae() {
        return nomeMae;
    }

    /**
     * @param nomeMae the nomeMae to set
     */
    public void setNomeMae(String nomeMae) {
        this.nomeMae = nomeMae;
    }

    public void setFone_cel(String Fone_cel) {
        this.Fone_cel = Fone_cel;
    }

}
