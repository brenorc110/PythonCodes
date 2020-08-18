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
public class FichaC {
    public static void main(String[] args) {
        Scanner e = new Scanner(System.in); 
        Paciente pc1 = new Paciente();
        System.out.println("Nome: ");
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
        
        System.out.println("Empresa de rabalho atual:");
        String trabalho = e.nextLine();
        System.out.println("Tempo de trabalho: ");
        int tempo_trabalho = e.nextInt();
        System.out.println("Filhos: 0, 1, 2,3 ,...  ");
        int num_filhos = e.nextInt();
        System.out.println("Religião?");
        String religiao = e.nextLine();
        System.out.println("Motivo da consulta: ");
        String motivo = e.nextLine();
        System.out.println("Já fez terapia antes? Se sim qual? ");
        String terapias_anteriores = e.nextLine();
        System.out.println("Faz exercícios físicos? Se sim qual? ");
        String exercs_fisicos = e.nextLine();
        System.out.println("Toma remédio frequente? Se sim, qual o objetivo? ");
        String remedio = e.nextLine();
        System.out.println("Como funciona o intestino? Regular?");
        String intestino = e.nextLine();
        System.out.println("Como funciona o aparelho digestivo? Regular?");
        String aparelho_digestivo = e.nextLine();
        System.out.println("Como funciona a menstruação? Regular?");
        String menstruacao = e.nextLine();
        System.out.println("OBS: Ver cor do olho, se fica ansioso, língua, olheiras, e postura.");
        String obs = e.nextLine();
        System.out.println("Responda com S ou N para os problemas abaixo: ");
        System.out.println("Artrite?");
        String artrite = e.nextLine();
        System.out.println("Bursite?");
        String bursite = e.nextLine();
        System.out.println("Câncer?");
        String cancer = e.nextLine();
        System.out.println("Tireoidismo?");
        String tireoidismo = e.nextLine();
        System.out.println("Tendinite?");
        String tendinite = e.nextLine();
        System.out.println("Nódulos?");
        String nodulos = e.nextLine();
        System.out.println("Febrite?");
        String febrite = e.nextLine();
        System.out.println("Traumatismos?");
        String traumatismos = e.nextLine();
        System.out.println("Escabiose?");
        String escabiose = e.nextLine();
        System.out.println("Artroses?");
        String artrose = e.nextLine();
        System.out.println("Dilatação da aorta?");
        String dil_aorta = e.nextLine();
        System.out.println("Neurite?");
        String neurite = e.nextLine();
        System.out.println("Tumor?");
        String tumor = e.nextLine();
        System.out.println("Osteoporose?");
        String oesteoporose = e.nextLine();
        System.out.println("Fibromialgia?");
        String fibromialgia = e.nextLine();
        System.out.println("Anemia?");
        String anemia = e.nextLine();
        System.out.println("FProblemas renais?");
        String prob_renais = e.nextLine();
        System.out.println("Hepatite?");
        String hepatite = e.nextLine();
        System.out.println("Alergia?");
        String alergia = e.nextLine();
        System.out.println("Outras?");
        String doencas_outros = e.nextLine();
        
        System.out.println("Conclusão do médico: ");
        String conclusao = e.nextLine();
        System.out.println("Tratamentos  a serem feitos (médico): ");
        String tratamentos = e.nextLine();
        System.out.println("Periodicidade (médico): ");
        String periodo = e.nextLine();
        System.out.println("OBS do médico: ");
        String obs_med = e.nextLine();
    }
}
