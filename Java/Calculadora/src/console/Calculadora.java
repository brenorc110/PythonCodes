package console;

import java.awt.Container;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.*;

public class Calculadora extends JFrame implements ActionListener{
    private static final long serialVersionUID = 1L;
    String op = "";
    private Float res;
    private Float n1;
    private Float n2;

    private JTextField txtResultado;
    private final JButton btnRes;
    private final JButton btnClr;
    private final JButton btnMais;
    private final JButton btnMenos;
    private final JButton btnVezes;
    private final JButton btnDividir;
    private final JButton btn0;
    private final JButton btn1;
    private final JButton btn2;
    private final JButton btn3;
    private final JButton btn4;
    private final JButton btn5;
    private final JButton btn6;
    private final JButton btn7;
    private final JButton btn8;
    private final JButton btn9;
    private final Container ctn;

@Override
public void actionPerformed(final ActionEvent e) {
    if (e.getActionCommand().equals("7")) {
        txtResultado.setText(txtResultado.getText() + "7");
    } else if (e.getActionCommand().equals("8")) {
        txtResultado.setText(txtResultado.getText() + "8");
    } else if (e.getActionCommand().equals("9")) {
        txtResultado.setText(txtResultado.getText() + "9");
    } else if (e.getActionCommand().equals("4")) {
        txtResultado.setText(txtResultado.getText() + "4");
    } else if (e.getActionCommand().equals("5")) {
        txtResultado.setText(txtResultado.getText() + "5");
    } else if (e.getActionCommand().equals("6")) {
        txtResultado.setText(txtResultado.getText() + "6");
    } else if (e.getActionCommand().equals("1")) {
        txtResultado.setText(txtResultado.getText() + "1");
    } else if (e.getActionCommand().equals("2")) {
        txtResultado.setText(txtResultado.getText() + "2");
    } else if (e.getActionCommand().equals("3")) {
        txtResultado.setText(txtResultado.getText() + "3");
    } else if (e.getActionCommand().equals("0")) {
        txtResultado.setText(txtResultado.getText() + "0");
    } else if (e.getActionCommand().equals("C")) {
        txtResultado.setText("");
    } else if (e.getActionCommand().equals("+")) {
        n1  = (float) Integer.parseInt(txtResultado.getText());
        txtResultado.setText("");
        op = "+";
    } else if (e.getActionCommand().equals("-")) {
        n1  = (float) Integer.parseInt(txtResultado.getText());
        txtResultado.setText("");
        op = "-";
    } else if (e.getActionCommand().equals("x")) {
        n1  = (float) Integer.parseInt(txtResultado.getText());
        txtResultado.setText("");
        op = "x";
    } else if (e.getActionCommand().equals("/")) {
        txtResultado.setText("");
        n1  = (float) Integer.parseInt(txtResultado.getText());
        op = "/";
    }

    if (e.getActionCommand().equals("=")) {
        switch (op) {
            case "+":
                n2  = (float) Integer.parseInt(txtResultado.getText());
                txtResultado.setText(String.valueOf(res));
                res = n1 + n2;
                txtResultado.setText(String.valueOf(res));
            case "-":
                n2  = (float) Integer.parseInt(txtResultado.getText());
                res = n1 - n2;
                txtResultado.setText(String.valueOf(res));
            case "x":
                n2  = (float) Integer.parseInt(txtResultado.getText());
                res = n1 * n2;
                txtResultado.setText(String.valueOf(res));
            case "/":
                n2  = (float) Integer.parseInt(txtResultado.getText());
                res = n1 / n2;
                txtResultado.setText(String.valueOf(res));
            default:
                break;
        }
        
    }
}

public Calculadora() {
    this.setSize(300, 500);
    this.setTitle("Calculadora");
    ctn = getContentPane();

    txtResultado = new JTextField("0");
    btnRes = new JButton("=");
    btnClr = new JButton("C");
    btnMais = new JButton("+");
    btnMenos = new JButton("-");
    btnVezes = new JButton("x");
    btnDividir = new JButton("/");
    btn7 = new JButton("7");
    btn8 = new JButton("8");
    btn9 = new JButton("9");
    btn4 = new JButton("4");
    btn5 = new JButton("5");
    btn6 = new JButton("6");
    btn1 = new JButton("1");
    btn2 = new JButton("2");
    btn3 = new JButton("3");
    btn0 = new JButton("0");

    txtResultado = new JTextField();

    btnRes.addActionListener(this);
    btnClr.addActionListener(this);
    btnMais.addActionListener(this);
    btnMenos.addActionListener(this);
    btnVezes.addActionListener(this);
    btnDividir.addActionListener(this);
    btn7.addActionListener(this);
    btn8.addActionListener(this);
    btn9.addActionListener(this);
    btn4.addActionListener(this);
    btn5.addActionListener(this);
    btn6.addActionListener(this);
    btn1.addActionListener(this);
    btn2.addActionListener(this);
    btn3.addActionListener(this);
    btn0.addActionListener(this);

    ctn.setLayout(null);

    ctn.add(txtResultado);
    ctn.add(btnRes);
    ctn.add(btnClr);
    ctn.add(btnMais);
    ctn.add(btnMenos);
    ctn.add(btnVezes);
    ctn.add(btnDividir);
    ctn.add(btn0);
    ctn.add(btn1);
    ctn.add(btn2);
    ctn.add(btn3);
    ctn.add(btn4);
    ctn.add(btn5);
    ctn.add(btn6);
    ctn.add(btn7);
    ctn.add(btn8);
    ctn.add(btn9);

    txtResultado.setBounds(20, 20, 250, 25);
    btn7.setBounds(20, 50, 50, 50);
    btn8.setBounds(80, 50, 50, 50);
    btn9.setBounds(140, 50, 50, 50);
    btnMais.setBounds(200, 50, 70, 50);
    btn4.setBounds(20, 110, 50, 50);
    btn5.setBounds(80, 110, 50, 50);
    btn6.setBounds(140, 110, 50, 50);
    btnMenos.setBounds(200, 110, 70, 50);
    btn1.setBounds(20, 180, 50, 50);
    btn2.setBounds(80, 180, 50, 50);
    btn3.setBounds(140, 180, 50, 50);
    btnVezes.setBounds(200, 180, 70, 50);
    btnDividir.setBounds(20, 240, 50, 50);
    btn0.setBounds(80, 240, 50, 50);
    btnClr.setBounds(140, 240, 50, 50);
    btnRes.setBounds(200, 240, 70, 50);

    // COLOCAR SEMPRE, SENÃO O APP FECHA MAS AINDA FICA EXECUTANDO.
    this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
}

public static void main(String[] args) {
    Calculadora c1 = new Calculadora();
        c1.setVisible(true);
    }
}


//public static void main(String[]args){
//        String op = null;
//        Int res;
//        String e1;
//        String e2;
//        double n1 = 0;
//        double n2 = 0;
//		
//	e1=JOptionPane.showInputDialog("Digite o primeiro número");
//        n1 = IntegerInteger.parseInt(e1);
//        	
//        op=JOptionPane.showInputDialog("Digite a operação solicitada; 1=Adição 2=Subtração 3=Mult 4=Divisão 5=Potenciação 6=Radiciação");
//        int i  = IntegerInteger.parseInt(op);
//         
//        e2=JOptionPane.showInputDialog("Digite o segundo número");
//        n2  = IntegerInteger.parseInt(e2);
//	
//            
//        /*if (op==null|| op.equals("") || op.equals("0")){
//			JOptionPane.showMessageDialog(null, "Por favor, insira o número da operação!");
//        op=JOptionPane.showInputDialog("Digite a operação solicitada; 1=Adição 2=Subtração 3=Mult 4=Divisão 5=Potenciação 6=Radiciação");
//		i  = IntegerInteger.parseInt(op);
//							
//        } else */
//		if (i>=7){
//        JOptionPane.showMessageDialog(null, "Por favor, insira o número da operação!");
//        op=JOptionPane.showInputDialog("Digite a operação solicitada; 1=Adição 2=Subtração 3=Mult 4=Divisão 5=Potenciação 6=Radiciação");
//		} else if (i<=0){
//        JOptionPane.showMessageDialog(null, "Por favor, insira o número da operação!");
//        op=JOptionPane.showInputDialog("Digite a operação solicitada; 1=Adição 2=Subtração 3=Mult 4=Divisão 5=Potenciação 6=Radiciação");
//		i  = IntegerInteger.parseInt(op);
//		}
//		switch (op) {
//                        case "1":
//                            res = n1+n2;
//                            JOptionPane.showMessageDialog(null, "O Resultado é: "+ res + "\n");
//                            break;
//                        case "2":
//                            res = n1-n2;
//                            JOptionPane.showMessageDialog(null, "O Resultado é: "+ res + "\n");
//                            break;
//                        case "3":
//                            res=n1*n2;
//                            JOptionPane.showMessageDialog(null, "O Resultado é: "+ res + "\n");
//                            break;
//                        case "4":
//                            res = n1/n2;
//                            JOptionPane.showMessageDialog(null, "O Resultado é: "+ res + "\n");
//                            break;
//                        case "5":
//                            res = Math.pow(n1,n2);
//                            JOptionPane.showMessageDialog(null, "O Resultado é: "+ res + "\n");
//                            break;
//                        default:
//                            res = Math.pow(n1,1.0/n2);
//                            JOptionPane.showMessageDialog(null, "O Resultado é: "+ res + "\n");
//                            break;
//							}
//		}							
//        
//		}        
//        //JOptionPane.showMessageDialog(null, "O resultado é: "+ op);
//        
        


//         n2  = IntegerInteger.parseInt(txtResultado.getText());
//         res = n1/n2;