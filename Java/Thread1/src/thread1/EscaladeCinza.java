
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
//para testar a classe procure uma imagem de no mínimo 30MB
// construa uma aplicação com threads para processar partes independentes da imagem

public class EscaladeCinza implements Runnable {
    public String NomeProcesso;
    @Override
    public void run() {
        System.out.println("start");
        // Inicia o buffer de img, vazio.
        BufferedImage image = null;
        try {
            //Abre imagem
            File imagefile = new File("C:\\TEMP\\a.jpeg");
            //Leia a imagem
            image = ImageIO.read(imagefile);

            //Varrendo imagem (altura e largura): 
            for (int x = 0; x < image.getWidth()/2; x++) {
                for (int y = 0; y < image.getHeight()/2; y++) {
                    System.out.println("programa "+ NomeProcesso + " x = " + x + "y = " + y);
                    //Conseguindo o RGB da imagem: 
                    Color c = new Color(image.getRGB(x, y));
                    //pegou cada canal e pôs na devida variável: 
                    int r = c.getRed();
                    int g = c.getGreen();
                    int b = c.getBlue();
                    //tirou a média: 
                    int media = (r + g + b) / 3;
                    //Especifica a nova imagem (colocando a média de tudo nos
                    //canais, vai dar a img em escala de cinza.
                    Color nc = new Color(media, media, media);
                    //Colocando os novos RGB.
                    image.setRGB(x, y, nc.getRGB());
                }
            }
            ImageIO.write(image, "jpg", new File("C:\\temp\\processamento.jpg"));
            //ImageIO.write(image, "bmp", new File("C:\\temp\\1.bmp"));
            //ImageIO.write(image, "png", new File("C:\\temp\\1.png"));

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        EscaladeCinza pa = new EscaladeCinza();
        pa.NomeProcesso = "Processo A";

        Thread t1 = new Thread(pa);
        t1.start();
        
        EscaladeCinza pb = new EscaladeCinza();
        pb.NomeProcesso = "Processo B";

        Thread t2 = new Thread(pb);
        t2.start();
    }

}
