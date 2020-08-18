/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package prova02;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JButton;
import javax.swing.JFrame;

public class Submarino extends JButton implements ActionListener{
    private int x;
    private int y;

    private boolean armado = false;
      
    public boolean isArmado(){
        return armado;
    }
    public Submarino(int x, int y) {
     this.x = x; this.y =y;  
     this.addActionListener(this);
     //this.setBounds(x*30, y*30, 30, 30);
        if(Math.random()>0.5){
            this.armado=true;
            System.out.println("ARMADO");
            this.setText("~");
        }else{
            System.out.println("DESARMADO");
            this.setText("~");
            
        }
    }   
    
    @Override
    public void actionPerformed(ActionEvent e) {
        System.out.print("oi");
        if (isArmado()== true){
              this.setText("*");  
              
            System.out.printf("DESTRUIU SUB (%d,%d)\n",this.x,this.y);
                   
        }else{
            this.setText("~");
            System.out.println("Acertou o Mar");
            
        }
          this.repaint();
        this.setEnabled(false);
      
        
    }
}
