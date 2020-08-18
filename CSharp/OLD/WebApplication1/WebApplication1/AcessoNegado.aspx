<%@ Page Title="" Language="C#" MasterPageFile="~/Site.Master" AutoEventWireup="true" CodeBehind="AcessoNegado.aspx.cs" Inherits="WebApplication1.AcessoNegado" %>
<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
</asp:Content>
<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <asp:Label ID="Label1" runat="server" 
        style="z-index: 1; left: 10px; top: 15px; position: absolute" 
        Text="Acesso Negado"></asp:Label>
    <asp:LinkButton ID="LinkButton1" runat="server" onclick="LinkButton1_Click" 
        style="z-index: 1; left: 10px; top: 40px; position: absolute">Voltar</asp:LinkButton>
</asp:Content>
