/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Lista3;

import java.util.Scanner;

/**
 *
 * @author fsj.5947373464
 */
public class FichaB {
    public static void main(String[] args) {
        Scanner e = new Scanner(System.in);
        Paciente pc1 = new Paciente();
        pc1.setNome(e.nextLine());
        System.out.println("Endereço (rua): ");
        pc1.setEndereco(e.nextLine());
        System.out.println("Bairro: ");
        pc1.setBairro(e.nextLine());
        System.out.println("CEP: ");
        pc1.setCEP(e.nextLine());
        System.out.println("Tel. Residencial: (xx) xxxx-xxxx. ");
        pc1.settelRes(e.nextLine());
        System.out.println("Tel. Comercial: (xx) xxxx-xxxx. ");
        pc1.settelCom(e.nextLine());
        System.out.println("Dia do nascimento: ");
        pc1.setdiaNasc(e.nextInt());
        System.out.println("Mês de nascimento: ");
        pc1.setmesNasc(e.nextInt());
        System.out.println("Ano de nascimento: ");
        pc1.setanoNasc(e.nextInt());
        System.out.println("Informe seu e-mail:");
        pc1.setEmail(e.nextLine());
        System.out.println("Profissão: ");
        pc1.setProfissao(e.nextLine());
        System.out.println("Como chegou a nós?");
        pc1.setIndicacao(e.nextLine());
        
        System.out.println("Qual a sua altura? \n");
        float alt = e.nextFloat();
        System.out.println("Qual o peso? \n");
        float peso = e.nextFloat();
        System.out.println("É diabética? S/N \n");
        String diab = e.nextLine();
        System.out.println("Está grávida? S/N \n");
        String grav = e.nextLine();
        System.out.println("É Cardíaca? S/N \n");
        String cardiaca = e.nextLine();
        System.out.println("Possui pressão alta? S/N \n");
        String p_alta = e.nextLine();
        System.out.println("Possui marcapassos? S/N \n");
        String marcapassos = e.nextLine();
        System.out.println("Usa anti-depressivos? S/N \n");
        String anti_depressivos = e.nextLine();
        System.out.println("Fuma? S/N \n");
        String fumante = e.nextLine();
        System.out.println("Usa analgésicos fortes? S/N \n");
        String analgesico = e.nextLine();
        System.out.println("Bebe? S/N \n");
        String alcool = e.nextLine();
        System.out.println("POssui doença crônica? S/N \n");
        String doenca_cronica = e.nextLine();
        System.out.println("Possui doença hereditária? S/N \n");
        String doenca_hereditaria = e.nextLine();
        System.out.println("POssui dores de cabeça fortes/ constantes? S/N \n");
        String dor_cabeca = e.nextLine();
        System.out.println("POssui dores de ossos? S/N \n");
        String dor_ossos = e.nextLine();
        System.out.println("POssui dores de articulações? S/N \n");
        String dor_articulacoes = e.nextLine();
        System.out.println("POssui dores de músculos? S/N \n");
        String dor_musculos= e.nextLine();
        System.out.println("Possui alguma outra dor? S/N \n");
        String dor_outros = e.nextLine();
        System.out.println("Realiza exercício físico? S/N \n");
        String exerc_fisicos = e.nextLine();
        System.out.println("Toma remédio frequente? Se sim, qual o objetivo? ");
        String remedio = e.nextLine();
        System.out.println("Como funciona o intestino? Regular?");
        String intestino = e.nextLine();
        System.out.println("Como funciona o aparelho digestivo? Regular?");
        String aparelho_digestivo = e.nextLine();
        System.out.println("Como funciona a menstruação? Regular?");
        String menstruacao = e.nextLine();
    }
    
}
