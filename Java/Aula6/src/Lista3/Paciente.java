/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Lista3;

import java.text.DateFormat;
import java.util.Calendar;
import java.util.Date;

/**
 *
 * @author fsj.5947373464
 */
public class Paciente {
    int ano, mes, dia;
        public static void main(String[] args) {
        int ano, mes, dia;
        Date d = new Date();
        Calendar c = Calendar.getInstance();
        c.setTime(new Date());
        
        c.setTime(d);
 
            DateFormat df = DateFormat.getDateInstance(DateFormat.SHORT);
            System.out.println(df.format(c.getTime()));
            
        }
  
   private String nome;
   public void setNome(String nome){
   this.nome = nome;
   }
   public String getNome(){
   return this.nome;
   }
   
   private String endereco;
   public void setEndereco(String endereco){
   this.nome = endereco;
   }
   public String getEndereco(){
   return this.endereco;
   }
   
   private String bairro;
   public void setBairro(String bairro){
   this.nome = bairro;
   }
   public String getBairro(){
   return this.bairro;
   }
   
   private String CEP;
   public void setCEP(String CEP){
   this.nome = CEP;
   }
   public String getCEP(){
   return this.CEP;
   }
   
   private String telRes;
   public void settelRes(String telRes){
   this.nome = telRes;
   }
   public String gettelRes(){
   return this.telRes;
   }
   
   private String telCom;
   public void settelCom(String telCom){
   this.nome = telCom;
   }
   public String gettelCom(){
   return this.telCom;
   }
   
   private int diaNasc;
   public void setdiaNasc(int diaNasc){
       
   this.diaNasc = diaNasc;
   }

    public int getdiaNasc(){
   return this.diaNasc;
   }
   
   private int mesNasc;
   public void setmesNasc(int mesNasc){
       
   this.mesNasc = mesNasc;
   }

    public int getmesNasc(){
   return this.mesNasc;
   }
    
    private int anoNasc;
   public void setanoNasc(int anoNasc){
       
   this.anoNasc = anoNasc;
   }

   public int getanoNasc(){
   return this.anoNasc;
   }
   
   private int idade;
   public int setIdade(int anoNasc){
       idade = anoNasc-Calendar.YEAR;
   return idade;
   }

   public int getIdade(int idade){
    idade = anoNasc-Calendar.YEAR;
   return this.idade;
   }
      
   private String email;
   public void setEmail(String email){
   this.email = email;
   }

    public String getEmail(){
   return this.email;
   }
   
   private String profissao;
   public void setProfissao(String profissao){
   this.profissao = profissao;
   }

    public String getProfissao(){
   return this.profissao;
   }
    
   private int tempoProfissao;
   public void settempoProfissao(int tempoProfissao){
   this.tempoProfissao = tempoProfissao;
   }

    public int gettempoProfissao(){
   return this.tempoProfissao;
   }
   
   private String indicacao;
   public void setIndicacao(String indicacao){
   this.indicacao = indicacao;
   }

    public String getIndicacao(){
   return this.indicacao;
   }
    
   
}
