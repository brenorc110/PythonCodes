/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Lista3;
import java.util.Scanner;
// CORREÇÃO PARA MODELO COM EMPREGO DA HERANÇA----------------------------------
public class AnamneseAuriculoterapia extends Paciente {

    public float getAlt() {
        return alt;
    }

    public void setAlt(float alt) {
        this.alt = alt;
    }

    public float getPeso() {
        return peso;
    }

    public void setPeso(float peso) {
        this.peso = peso;
    }

    public String getDiab() {
        return diab;
    }

    public void setDiab(String diab) {
        this.diab = diab;
    }

    public String getGrav() {
        return grav;
    }

    public void setGrav(String grav) {
        this.grav = grav;
    }

    public String getCardiaca() {
        return cardiaca;
    }

    public void setCardiaca(String cardiaca) {
        this.cardiaca = cardiaca;
    }

    public String getP_alta() {
        return p_alta;
    }

    public void setP_alta(String p_alta) {
        this.p_alta = p_alta;
    }

    public String getMarcapassos() {
        return marcapassos;
    }

    public void setMarcapassos(String marcapassos) {
        this.marcapassos = marcapassos;
    }

    public String getAnti_depressivos() {
        return anti_depressivos;
    }

    public void setAnti_depressivos(String anti_depressivos) {
        this.anti_depressivos = anti_depressivos;
    }

    public String getFumante() {
        return fumante;
    }

    public void setFumante(String fumante) {
        this.fumante = fumante;
    }

    public String getAnalgesico() {
        return analgesico;
    }

    public void setAnalgesico(String analgesico) {
        this.analgesico = analgesico;
    }

    public String getAlcool() {
        return alcool;
    }

    public void setAlcool(String alcool) {
        this.alcool = alcool;
    }

    public String getDoenca_cronica() {
        return doenca_cronica;
    }

    public void setDoenca_cronica(String doenca_cronica) {
        this.doenca_cronica = doenca_cronica;
    }

    public String getDoenca_hereditaria() {
        return doenca_hereditaria;
    }

    public void setDoenca_hereditaria(String doenca_hereditaria) {
        this.doenca_hereditaria = doenca_hereditaria;
    }

    public String getDor_cabeca() {
        return dor_cabeca;
    }

    public void setDor_cabeca(String dor_cabeca) {
        this.dor_cabeca = dor_cabeca;
    }

    public String getDor_ossos() {
        return dor_ossos;
    }

    public void setDor_ossos(String dor_ossos) {
        this.dor_ossos = dor_ossos;
    }

    public String getDor_musculos() {
        return dor_musculos;
    }

    public void setDor_musculos(String dor_musculos) {
        this.dor_musculos = dor_musculos;
    }

    public String getExerc_fisicos() {
        return exerc_fisicos;
    }

    public void setExerc_fisicos(String exerc_fisicos) {
        this.exerc_fisicos = exerc_fisicos;
    }

    public String getFrequencia_agua() {
        return frequencia_agua;
    }

    public void setFrequencia_agua(String frequencia_agua) {
        this.frequencia_agua = frequencia_agua;
    }

    public String getEstresse() {
        return estresse;
    }

    public void setEstresse(String estresse) {
        this.estresse = estresse;
    }

    public String getAlimentacao() {
        return alimentacao;
    }

    public void setAlimentacao(String alimentacao) {
        this.alimentacao = alimentacao;
    }

    public String getQualidade_alimentacao() {
        return qualidade_alimentacao;
    }

    public void setQualidade_alimentacao(String qualidade_alimentacao) {
        this.qualidade_alimentacao = qualidade_alimentacao;
    }

    public String getRemedio() {
        return remedio;
    }

    public void setRemedio(String remedio) {
        this.remedio = remedio;
    }
    public String getDor_outros() {
        return dor_outros;
    }

    public void setDor_outros(String dor_outros) {
        this.dor_outros = dor_outros;
    }
    private String dor_articulacoes;

    public String getDor_articulacoes() {
        return dor_articulacoes;
    }

    public void setDor_articulacoes(String dor_articulacoes) {
        this.dor_articulacoes = dor_articulacoes;
    }
    
    private float alt;
    private float peso;
    private String diab;
    private String grav;
    private String cardiaca;
    private String p_alta;
    private String marcapassos;
    private String anti_depressivos;
    private String fumante;
    private String analgesico;
    private String alcool;
    private String doenca_cronica;
    private String doenca_hereditaria;
    private String dor_cabeca;
    private String dor_outros;
    private String dor_ossos;
    private String dor_musculos;
    private String exerc_fisicos;
    private String frequencia_agua;
    private String estresse;
    private String alimentacao;
    private String qualidade_alimentacao;
    private String remedio;

    public AnamneseAuriculoterapia() {
    }
    public static void main(String[] args) {
        Scanner e = new Scanner(System.in);    
        AnamneseAuriculoterapia fa = new AnamneseAuriculoterapia();
        fa.setNome(e.nextLine());
        System.out.println("Endereço (rua): ");
        fa.setEndereco(e.nextLine());
        System.out.println("Bairro: ");
        fa.setBairro(e.nextLine());
        System.out.println("CEP: ");
        fa.setCEP(e.nextLine());
        System.out.println("Tel. Residencial: (xx) xxxx-xxxx. ");
        fa.settelRes(e.nextLine());
        System.out.println("Tel. Comercial: (xx) xxxx-xxxx. ");
        fa.settelCom(e.nextLine());
        System.out.println("Dia do nascimento: ");
        fa.setdiaNasc(e.nextInt());
        System.out.println("Mês de nascimento: ");
        fa.setmesNasc(e.nextInt());
        System.out.println("Ano de nascimento: ");
        fa.setanoNasc(e.nextInt());
        System.out.println("Informe seu e-mail:");
        fa.setEmail(e.nextLine());
        System.out.println("Profissão: ");
        fa.setProfissao(e.nextLine());
        System.out.println("Como chegou a nós?");
        fa.setIndicacao(e.nextLine());
        
        System.out.println("Qual a sua altura? \n");
        fa.setAlt(e.nextFloat());
        System.out.println("Qual o peso? \n");
        fa.setPeso(e.nextFloat());
        System.out.println("É diabética? S/N \n");
        fa.setDiab(e.nextLine());
        System.out.println("Está grávida? S/N \n");
        fa.setGrav(e.nextLine());
        System.out.println("É Cardíaca? S/N \n");
        fa.setCardiaca (e.nextLine());
        System.out.println("Possui pressão alta? S/N \n");
        fa.setP_alta(e.nextLine());
        System.out.println("Possui marcapassos? S/N \n");
        fa.setMarcapassos(e.nextLine());
        System.out.println("Usa anti-depressivos? S/N \n");
        fa.setAnti_depressivos(e.nextLine());
        System.out.println("Fuma? S/N \n");
        fa.setFumante(e.nextLine());
        System.out.println("Usa analgésicos fortes? S/N \n");
        fa.setAnalgesico (e.nextLine());
        System.out.println("Bebe? S/N \n");
        fa.setAlcool (e.nextLine());
        System.out.println("POssui doença crônica? S/N \n");
        fa.setDoenca_cronica(e.nextLine());
        System.out.println("Possui doença hereditária? S/N \n");
        fa.setDoenca_hereditaria(e.nextLine());
        System.out.println("POssui dores de cabeça fortes/ constantes? S/N \n");
        fa.setDor_cabeca(e.nextLine());
        System.out.println("POssui dores de ossos? S/N \n");
        fa.setDor_ossos (e.nextLine());
        System.out.println("POssui dores de articulações? S/N \n");
        fa.setDor_articulacoes (e.nextLine());
        System.out.println("POssui dores de músculos? S/N \n");
        fa.setDor_musculos(e.nextLine());
        System.out.println("Possui alguma outra dor? S/N \n");
        fa.setDor_outros (e.nextLine());
        System.out.println("Realiza exercício físico? S/N \n");
        fa.setExerc_fisicos(e.nextLine());
        System.out.println("Bebe bastante água? S/N \n");
        fa.setFrequencia_agua(e.nextLine());
        System.out.println("Considera vida estressante? S/N \n");
        fa.setEstresse (e.nextLine());
        System.out.println("Alimentação? Pouca, razoável, muita, mais do que deveria. ");
        fa.setAlimentacao (e.nextLine());
        System.out.println("Alimentação? Mal, regular, boa, muito bem. ");
        fa.setQualidade_alimentacao(e.nextLine());
        System.out.println("Toma remédio frequente? Se sim, qual o objetivo? ");
        fa.setRemedio(e.nextLine());
    }
}
