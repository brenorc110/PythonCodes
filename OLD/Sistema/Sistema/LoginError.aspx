<%@ Page Title="" Language="C#" MasterPageFile="~/MasterPages/Login.Master" AutoEventWireup="true" CodeBehind="LoginError.aspx.cs" Inherits="Sistema.LoginError" %>
<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
</asp:Content>
<asp:Content ID="Content2" ContentPlaceHolderID="MainPlaceHolder" runat="server">
    <div class="alert alert-block">
        <strong>Acesso Negado!</strong> Você não possui acesso ao sistema, por favor, <a href="Login.aspx">tente novamente</a>.
    </div>
</asp:Content>
