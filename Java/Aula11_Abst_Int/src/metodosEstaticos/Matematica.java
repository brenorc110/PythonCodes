/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package metodosEstaticos;

import java.util.Scanner;

public class Matematica {

    private static int used = 0;

public static double tangente(double angulo){
  return Math.tan(angulo);
}
public static double cosseno(double angulo){
  return Math.cos(angulo);
}
public static double seno(double angulo){
    used++;
  return Math.sin(angulo);
}
public static double areaQuadrado(double lado){
    used++;
    return lado*lado;
}
public static double areaRetangulo(double base, double altura){
    used++;
return base*altura;
}
public static double areaLosango(double base, double altura){
    used++;
return base*altura;
}
public static double areaTriangulo(double base, double altura){
    used++;
return (base*altura)/2;
}
public static double areaPentagono(double base, double altura){
    used++;
    return 5*((base*altura)/2);
}
public static double areaHexagono(double base, double altura){
    used++;
    return 6*((base*altura)/2);
}
public static double areaCircunferencia(double raio){
    used++;
return 3.141*(Math.pow(raio, 2));
}
public static double areaTetraedro(double altura){
    used++;
return (Math.pow(altura,2))*1.732;
}
public static double areaOctaedro(double altura){
    used++;
return 2*(Math.pow(altura,2))*1.732;
}
public static double areaCubo(double altura){
    used++;
return 6*(Math.pow(altura,2));
}
public static double areaDodecaedro(double altura, double ap){
    used++;
return 30*altura*ap;
}
public static double areaIcosaedro(double altura){
    used++;
return (5*(Math.pow(altura,2))*1.732);
}
public static double perimetroQuadrado(double lado){
return 4*lado;
}
public static double perimetroRetangulo(double base, double altura){
return (2*base)+(2*altura);
}
public static double perimetroTriangulo(double h, double ca, double cop){
    used++;
return h+ca+cop;
}
public static double perimetroRombo(double lado){
    used++;
return 4*lado;
}
public static double perimetroParalelogramo(double base, double altura){
    used++;
return (2*base)+(2*altura);
}
public static double perimetroTrapezio(double a, double b, double c, double d){
    used++;
return a+b+c+d;
}
public static double perimetroPoligono(double numLados, double a){
    used++;
return a*numLados;
}
public static double perimetroCircunferencia(double raio){
    used++;
return 2*3.141*raio;
}


    public static void main(String[] args) {
		int op, op2 = 1;
        double angulo, ap, lado, raio, a, b, c, d, base, altura, h, ca, cop, numLados;;
        Scanner e = new Scanner(System.in);
		
		do{
		System.out.println("- Bem vindo ao programa de figuras geometricas: ");
		System.out.println("Escolha a opcao desejada: ");
		System.out.println("1 - Area e perimetro do quadrado: ");
		System.out.println("2 - Area e perimetro do retangulo: ");
		System.out.println("3- Area do losango: ");
		System.out.println("4- Seno, Cosseno e tangente de angulo: ");
		System.out.println("5- Area do triangulo: ");
		System.out.println("6- Perimetro do triangulo: ");
		System.out.println("7- area do pentagono: ");
		System.out.println("8- area do hexagono: ");
		System.out.println("9- area e perimetro da circunferencia: ");
		System.out.println("10- Area do tetraedro: ");
		System.out.println("11- area do octaedro: ");
		System.out.println("12- Area do cubo: ");
		System.out.println("13- Area do dodecaedro: ");
		System.out.println("14- Area do icosaedro: ");
		System.out.println("15- Perimetro do rombo:  ");
		System.out.println("16- Perimetro do paralelogramo: ");
		System.out.println("17- Perimetro do trapezio: ");
		System.out.println("18- Perimetro do poligono regular: ");
		op = e.nextInt();
		
		switch(op){
		case 1:
		System.out.println("- Para calculo da area e perimetro do quadrado: ");
		System.out.println("Informe o valor do lado: ");
		lado = e.nextDouble();
		System.out.println("areaQuadrado= " + Matematica.areaQuadrado(lado));
		System.out.println("perimetroQuadrado= " + Matematica.perimetroQuadrado(lado));
		break;
                
		case 2:
		System.out.println("- Para calculo da area e perimetro do retangulo: ");
		System.out.println("Informe o valor da base: ");
		base = e.nextDouble();
		System.out.println("Informe o valor da altura: ");
		altura = e.nextDouble();
		System.out.println("areaRetangulo= "+ Matematica.areaRetangulo(base, altura));
		System.out.println("perimetroRetangulo= "+ Matematica.perimetroRetangulo(base, altura));
		break;
                
		case 3:
		System.out.println("- Para calculo da area do losango: ");
		System.out.println("Informe o valor da base: ");
		base = e.nextDouble();
		System.out.println("Informe o valor da altura: ");
		altura = e.nextDouble();
		System.out.println("areaLosango= "+ Matematica.areaLosango(base, altura));
		break;
                
		case 4:
		System.out.println("- Para calculo de seno, cosseno e tangente: ");
            System.out.println("Informe o valor do ângulo: ");
            angulo = e.nextDouble();
		 System.out.println("Seno = "+ Matematica.seno(angulo));
            System.out.println("Cosseno = "+ Matematica.cosseno(angulo));
            System.out.println("Tangente = "+ Matematica.tangente(angulo));
		break;
                
		case 5:
		System.out.println("- Para calculo da area do triangulo: ");
		System.out.println("Informe o valor da base: ");
		base = e.nextDouble();
		System.out.println("Informe o valor da altura: ");
		altura = e.nextDouble();
		System.out.println("areaTriangulo= "+ Matematica.areaTriangulo(base, altura));
		break;
                
		case 6:
		System.out.println("- Para calculo do perimetro do triangulo: ");
		System.out.println("Informe o valor da hipotenusa: ");
		h = e.nextDouble();
		System.out.println("Informe o valor do cateto adjacente: ");
		ca = e.nextDouble();
		System.out.println("Informe o valor do cateto oposto: ");
		cop = e.nextDouble();
		System.out.println("perimetroTriangulo= "+ Matematica.perimetroTriangulo(h, ca, cop));
		break;
                
		case 7:
		System.out.println("- Para calculo da area do pentagono: ");
		System.out.println("Informe o valor da base: ");
		base = e.nextDouble();
		System.out.println("Informe o valor da altura: ");
		altura = e.nextDouble();
		System.out.println("areaPentagono= "+ Matematica.areaPentagono(base, altura));
		break;
                
		case 8:
		System.out.println("- Para calculo da area do hexagono: ");
		System.out.println("Informe o valor da base: ");
		base = e.nextDouble();
		System.out.println("Informe o valor da altura: ");
		altura = e.nextDouble();
		System.out.println("areaHexagono= "+ Matematica.areaHexagono(base, altura));
		break;
                
		case 9:
		System.out.println("- Para calculo da area e perimetro da circunferencia: ");
		System.out.println("Informe o valor do raio: ");
		raio = e.nextDouble();
		System.out.println("areaCircunferencia= "+ Matematica.areaCircunferencia(raio));
		System.out.println("perimetroCircunferencia= "+ Matematica.perimetroCircunferencia(raio));
		break;
                
		case 10:
		System.out.println("- Para calculo da area do tetraedro: ");
		System.out.println("Informe o valor da altura: ");
		altura = e.nextDouble();
		System.out.println("areaTetraedro= " + Matematica.areaTetraedro(altura));
		break;
                
		case 11:
		System.out.println("- Para calculo da area do octaedro: ");
		System.out.println("Informe o valor da altura: ");
		altura = e.nextDouble();
		System.out.println("areaOctaedro= " + Matematica.areaOctaedro(altura));
		break;
                
		case 12:
		System.out.println("- Para calculo da area do cubo: ");
		System.out.println("Informe o valor da altura: ");
		altura = e.nextDouble();
		System.out.println("areaCubo= " + Matematica.areaCubo(altura));
		break;
                
		case 13:
		System.out.println("- Para calculo da area do dodecaedro: ");
		System.out.println("Informe o valor da altura: ");
		altura = e.nextDouble();
		System.out.println("Informe o valor da apotema: ");
		ap = e.nextDouble();
		System.out.println("areaDodecaedro= " + Matematica.areaDodecaedro(altura, ap));
		break;
                
		case 14:
		System.out.println("- Para calculo da area do icosaedro: ");
		System.out.println("Informe o valor da altura: ");
		altura = e.nextDouble();
		System.out.println("areaIcosaedro= " + Matematica.areaIcosaedro(altura));
		break;
                
		case 15:
		System.out.println("- Para calculo do perimetro do rombo: ");
		System.out.println("Informe o valor do lado: ");
		lado = e.nextDouble();
		System.out.println("perimetroRombo= " + Matematica.perimetroRombo(lado));
		break;
                
		case 16:
		System.out.println("- Para calculo do perimetro do paralelogramo: ");
		System.out.println("Informe o valor da base: ");
		base = e.nextDouble();
		System.out.println("Informe o valor da altura: ");
		altura = e.nextDouble();
		System.out.println("perimetroParalelogramo= "+ Matematica.perimetroParalelogramo(base, altura));
		break;
                
		case 17:
		System.out.println("- Para calculo do perimetro do trapezio: ");
		System.out.println("Informe o valor do lado A: ");
		a = e.nextDouble();
		System.out.println("Informe o valor do lado B: ");
		b = e.nextDouble();
		System.out.println("Informe o valor do lado C: ");
		c = e.nextDouble();
		System.out.println("Informe o valor do lado D: ");
		d = e.nextDouble();
		System.out.println("perimetroTrapezio= "+ Matematica.perimetroTrapezio(a, b, c, d));
		break;
                
		case 18:
		System.out.println("- Para calculo do perimetro do poligono regular: ");
		System.out.println("Informe o numero de lados que o mesmo possui: ");
		numLados = e.nextDouble();
		System.out.println("Informe o valor do lado A: ");
		a = e.nextDouble();
		System.out.println("perimetroPoligono= "+ Matematica.perimetroPoligono(numLados, a));
                break;
                
		default:
		System.out.println("Opção incorreta!");
		break;
		}
		
		System.out.println("Quer fazer outra operacao? 1=Sim/2=Não");
		op2 = e.nextInt();
		} while (op2 == 1);
		
		System.out.println("Numero de vezes utilizado= " + used);
    }
}
