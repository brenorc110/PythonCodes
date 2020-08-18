package aula5;
/**
 * POLIMORFISMO E SOBRECARGA
 * @author fsj.5947373464
 */
public class Ponto2D {
    private double x;
    private double y;
    Ponto2D(){
        System.out.println("Construtor vazio.");
    }
    Ponto2D(double x, double y){
        this.x = x;
        this.y = y;
        System.out.println("Construtor double.");
    }
    Ponto2D(int x, int y){
        this.x = x;
        this.y = y;
        System.out.println("Construtor int.");
    }
    Ponto2D(float x, float y){
        this.x = x;
        this.y = y;
        System.out.println("Construtor float"); 
    }
    public double CalcularDistancia(Ponto2D p1, Ponto2D p2){
        
        return Math.sqrt(Math.pow(p1.y-p1.x,2) + Math.pow(p2.y-p2.x,2));
    }
    public double CalcularDistancia(double x2, double x1, double y2, double y1){
   
        return Math.sqrt(Math.pow(x2-x1,2) + Math.pow(y2-y1,2));
    }
    public static void main(String[] args) {
        Ponto2D p = new Ponto2D();
        /*
        p = new Ponto2D(1.1, 1);
        p = new Ponto2D(1.2f, 1);
         */
        Ponto2D p1 = new Ponto2D(5,3);
        Ponto2D p2 = new Ponto2D(2,1);
        
        System.out.println(p.CalcularDistancia(p1, p2));
        System.out.println(p.CalcularDistancia(5, 3, 2, 1));
    }
    
}
