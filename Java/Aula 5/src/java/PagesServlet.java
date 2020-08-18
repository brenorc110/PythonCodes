import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

@WebServlet(urlPatterns = {"/PagesServlet"})
public class PagesServlet extends HttpServlet {
@Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        HttpSession session = request.getSession(true);
    
        response.setContentType("text/html;charset=UTF-8");
        PrintWriter out = response.getWriter();
        try {
           if (session.getAttribute("ID") == "2134") {
           out.println("<!DOCTYPE html>");
           out.println("<html>");
           out.println("<head>");
           out.println("<title>Página Restrita</title>"); 
           out.println("</head>");
           out.println("<body>");
           out.println("Bem vindo " + session.getAttribute("EMAIL") + "</br>");
           out.println("Sua Sessão é: " + session.getAttribute("ID"));
           out.println("</br>");
               if (session.getAttribute("EMAIL").equals("joao.silva@meuemail.com")) {
                   out.println("<ul>");
                   out.println("<li>Menu 2</li>");
                   out.println("<li>Menu 4</li>");
                   out.println("</ul>");
               }
               if (session.getAttribute("EMAIL").equals("maria.silveira@meuemail.com")) {
                   out.println("<ul>");
                   out.println("<li>Menu 1</li>");
                   out.println("<li>Menu 2</li>");
                   out.println("<li>Menu 3</li>");
                   out.println("<li>Menu 4</li>");
                   out.println("<li>Menu 5</li>");
                   out.println("</ul>");
               }
             
           out.println("</body>");
           out.println("</html>"); 
            }
           else{
           out.println("<!DOCTYPE html>");
           out.println("<html>");
           out.println("<head>");
           out.println("<title>Página de Erro</title>"); 
           out.println("</head>");
           out.println("<body>");
           out.println("<h1>Espera ae espertinho! sua sessão não está correta!</h1>");
           out.println("<p>A sua ID de sessão é " + session.getAttribute("ID") + " mas deveria ser 2134!</p>");
           out.println("<a href=\"LoginServlet\">Volte e faça seu Login direito!</a>");
           out.println("</body>");
           out.println("</html>"); 
            }
            
                    
        } finally {            
            out.close();
        }
    }
}
