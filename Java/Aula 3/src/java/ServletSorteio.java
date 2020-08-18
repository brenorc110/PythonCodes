/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 *
 * @author ALUNOS
 */
@WebServlet(name = "ServletSorteio", urlPatterns = {"/ServletSorteio"})
public class ServletSorteio extends HttpServlet {

    /**
     * Processes requests for both HTTP
     * <code>GET</code> and
     * <code>POST</code> methods.
     *
     * @param request servlet request
     * @param response servlet response
     * @throws ServletException if a servlet-specific error occurs
     * @throws IOException if an I/O error occurs
     */
    protected void processRequest(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        response.setContentType("text/html;charset=UTF-8");
        PrintWriter out = response.getWriter();
        try {
            Random aleatorio = new Random();
         
            out.println("<!DOCTYPE html>");
            out.println("<html>");
            out.println("<head><title>Nome da Página</title></head>");
            out.println("<body>"
                    + "<h1>Mega-Sena</h1>"
                    + "<h2>Números Sorteados</h2>"
                    + "Primeiro Número: "+aleatorio.nextInt(61) + "</br>"
                    + "Segundo Número: "+aleatorio.nextInt(61) + "</br>"
                    + "Terceiro Número: "+aleatorio.nextInt(61) + "</br>"
                    + "Quarto Número: "+aleatorio.nextInt(61) + "</br>"
                    + "Quinto Número: "+aleatorio.nextInt(61) + "</br>"
                    + "Sexto Número: "+aleatorio.nextInt(61) + "</br>"
                    + "</body>");
            out.println("</html>");
        } finally {            
            out.close();
        }
    }

    // <editor-fold defaultstate="collapsed" desc="HttpServlet methods. Click on the + sign on the left to edit the code.">
    /**
     * Handles the HTTP
     * <code>GET</code> method.
     *
     * @param request servlet request
     * @param response servlet response
     * @throws ServletException if a servlet-specific error occurs
     * @throws IOException if an I/O error occurs
     */
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        processRequest(request, response);
    }

    /**
     * Handles the HTTP
     * <code>POST</code> method.
     *
     * @param request servlet request
     * @param response servlet response
     * @throws ServletException if a servlet-specific error occurs
     * @throws IOException if an I/O error occurs
     */
    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        processRequest(request, response);
    }

    /**
     * Returns a short description of the servlet.
     *
     * @return a String containing servlet description
     */
    @Override
    public String getServletInfo() {
        return "Short description";
    }// </editor-fold>
}
