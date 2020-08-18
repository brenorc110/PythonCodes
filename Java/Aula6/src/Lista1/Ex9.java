package Lista1;

import java.util.Scanner;

public class Ex9{

    private static int i, n;

    private static int divisivelpor3ou5(int i) {
        if (i%3==0 || i%5==0) return 1; else return 0;
    }
 
 public static void main(String[] args) {
  Scanner e = new Scanner(System.in);

    System.out.println("O numero: ");
    n = e.nextInt();
    for (i=0; i<=n; i++){
      divisivelpor3ou5(i);
      
      if(divisivelpor3ou5(i)==1 && i!=0) System.out.println(i);
    }
}
}