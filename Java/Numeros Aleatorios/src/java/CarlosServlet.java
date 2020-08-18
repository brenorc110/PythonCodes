
import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.util.Random;


@WebServlet(name = "CarlosServlet", urlPatterns = {"/CarlosServlet"})
public class CarlosServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {        
     
        Random gerador = new Random();
 
        int numero = gerador.nextInt(10)+ 10;
 
        System.out.println(numero);
           
        Random gerador1 = new Random();
 
        int numero1 = gerador.nextInt(20)+ 10;
 
        System.out.println(numero);
        
        Random gerador2 = new Random();
 
        int numero2 = gerador.nextInt(30)+ 10;
 
        System.out.println(numero);
        
        Random gerador3 = new Random();
 
        int numero3 = gerador.nextInt(40)+ 10;
 
        System.out.println(numero);
        
        Random gerador4 = new Random();
 
        int numero4 = gerador.nextInt(50)+ 10;
 
        System.out.println(numero);
        
        Random gerador5 = new Random();
 
        int numero5 = gerador.nextInt(20)+ 10;
 
        System.out.println(numero);
        
        response.setContentType("text/html;charset=UTF-8");
        PrintWriter out = response.getWriter();
        try {
            out.println("<html>");
            out.println("<head>");
            out.println("<title>Servlet CarlosServlet</title>");            
            out.println("</head>");
            out.println("<body>");
            out.println("<h1>" + numero + "</h1>");
            out.println("<h1>" + numero1 + "</h1>");
            out.println("<h1>" + numero2 + "</h1>");
            out.println("<h1>" + numero3 + "</h1>");
            out.println("<h1>" + numero4 + "</h1>");
            out.println("<h1>" + numero5 + "</h1>");
            out.println("</body>");
            out.println("</html>");
        } finally {            
            out.close();
        }
    }
}
