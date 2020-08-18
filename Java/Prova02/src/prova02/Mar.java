/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package prova02;

import java.awt.Container;
import javax.swing.JFrame;
import javax.swing.JButton;

/**
 *
 * @author fsj.5947373464
 */
public class Mar extends JFrame {

    Mar() {
        setSize(500, 500);
        Container ctn = getContentPane();
        ctn.setLayout(null);
        for (int x = 0; x <= 400; x += 51) {
            for (int y = 0; y <= 400; y += 51) {
                Submarino btnok = new Submarino(x,y);
                btnok.setBounds(x, y, 50, 50);
                ctn.add(btnok);
                  ctn.repaint();
            }
        }
        this.setVisible(true);
        
        // System.exit(0);
    }

    public static void main(String[] args) {
        Mar p2 = new Mar();
    }

}
