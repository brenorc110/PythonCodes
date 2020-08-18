package thread1;

import java.util.Scanner;
public class Monitor implements Runnable{
	private Thread th;
	private boolean monitorando;
	
	public Monitor(){
		monitorando=true;
		th=new Thread(this);
	}
	public void iniciar(){
		th.start();
	}
	public void parar(){
		monitorando=false;
		System.out.println("Interrompendo...");
		th.interrupt();
		try{
			th.join(2000);
		} catch(InterruptedException e){
			e.printStackTrace();
		}
	}
	public void run(){
		System.out.println("Iniciando monitoramento. ");
		while(monitorando==true){
			    System.out.println("Monitorando");
		try{
			Thread.sleep(1000);
		} catch (InterruptedException e){
			monitorando=false;
		}
	}
	}
	
	public static void main(String[] args){
		Monitor monitor = new Monitor();
		monitor.iniciar();
		Scanner sc = new Scanner(System.in);
		boolean monitorar = true;
		do{
		    System.out.println("Continuar monitoramento? S/N");
		    String resp = sc.next();
		    if(resp.equalsIgnoreCase("N")== true){
		        monitorar=false;
		        monitor.parar();
		   } else{
		       System.out.println("Monitorando...");
		   }
		} while(monitorar==true);
	    sc.close();
	}
	
}