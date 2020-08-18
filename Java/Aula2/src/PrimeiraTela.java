/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
import java.awt.Container; 
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import javax.swing.*; 
import javax.swing.event.DocumentEvent; 
import javax.swing.event.DocumentListener;
        
/**
 * Aplicação de formulário e eventos.
 * @author breno
 */
public class PrimeiraTela extends JFrame implements ActionListener{
    private JButton btnOK;
    private JTextField txtNome; 
    private JTextField txtSobrenome; 
    private JLabel lblNome; 
    private JLabel lblSobrenome;
    private Container ctn; 
    
    public void trataBotaoOK(){
        JOptionPane.showMessageDialog(null,"Botao 2 clicado");
    }
    public PrimeiraTela(){
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(400,300);
        setTitle("Eventos em Java Swing");
        ctn = getContentPane();
        ctn.setLayout(null);
        btnOK = new JButton("Enviar");
        btnOK.addActionListener(this);
        /*
        btnOK.addActionListener(new ActionListener(){
            @Override
            public void actionPerformed(ActionEvent e){
                trataBotaoOK();
        }
        });
        */
        lblNome = new JLabel("Nome: ");  
        txtNome = new JTextField();
        lblNome.setBounds(10,10,100,25);
        txtNome.setBounds(70,10,200,25); 
        txtNome.addFocusListener(new FocusListener() {
            @Override
            public void focusLost(FocusEvent e){

            }
            @Override
            public void focusGained(FocusEvent e){

            }});
        txtNome.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void removeUpdate(DocumentEvent e){
                JOptionPane.showMessageDialog(null, "Update");
            }
            @Override
            public void insertUpdate(DocumentEvent e){
                //Quando texto for inserido:
                JOptionPane.showMessageDialog(null, "Insert");
            }
            @Override
            public void changedUpdate(DocumentEvent e){
                //Quando texto for alterado: 
                JOptionPane.showMessageDialog(null, "changed");
            }
        });
        lblSobrenome = new JLabel("Sobrenome: ");
        lblSobrenome.setBounds(10, 40, 100, 25);
        txtSobrenome = new JTextField();
        txtSobrenome.setBounds(80, 40, 200, 25);
        txtSobrenome.addFocusListener(new FocusListener() {
            @Override
            public void focusLost(FocusEvent e){
                //JOptionPane.showMessageDialog(null, "Foco perdido!");
            }
            @Override
            public void focusGained(FocusEvent e){
                //JOptionPane.showMessageDialog(null, "Foco adquirido");
            }});
        btnOK = new JButton("OK");
        btnOK.setBounds(110, 80, 80, 40);
        ctn.add(txtSobrenome);
        ctn.add(lblNome);
        ctn.add(lblSobrenome);
        ctn.add(txtNome);
        ctn.add(btnOK);
}
    
    public static void main(String[] args) {
        PrimeiraTela tela = new PrimeiraTela();
        tela.setVisible(true);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if(e.getActionCommand().equals("Enviar")) {  
            JOptionPane.showMessageDialog(null, "Botao clicado 1");
        }
    }
}
