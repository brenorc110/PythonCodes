

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Calendar;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;


@WebServlet(name = "MeuServlet", urlPatterns = {"/MeuServlet"})
public class MeuServlet extends HttpServlet {

   @Override
    protected void doGet (HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        Calendar hoje = Calendar.getInstance();
        int hora = hoje.get(Calendar.HOUR_OF_DAY);
        String mensagem = "";
        
        if (hora >= 0 && hora <= 12) {
            mensagem = "Bom Dia!";
        } else if (hora >= 12 && hora <=18) {
            mensagem = "Boa Tarde!";
        }  else if (hora >= 18 && hora <=23) {
            mensagem = "Boa Noite!";
        }
        response.setContentType("text/html;charset=UTF-8");
        PrintWriter out = response.getWriter();
        try {
           
            out.println("<! DOCTYPE html>");
            out.println("<html>");
            out.println("<title>Servlet MeuServlet</title>");            
            out.println("</head><title>Página dia</title></head>");
            out.println("<body><h1>" + mensagem + "</h1></body>");
            out.println("</html>");
        } finally {            
            out.close();
        }
    }

}