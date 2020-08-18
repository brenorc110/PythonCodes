<%@ Page Title="" Language="C#" MasterPageFile="~/Site.Master" AutoEventWireup="true" CodeBehind="Login.aspx.cs" Inherits="WebApplication1.Login" %>
<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
</asp:Content>
<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <asp:Label ID="lbEmail" runat="server" 
        style="z-index: 1; left: 10px; top: 15px; position: absolute" Text="E-mail:"></asp:Label>
    <asp:TextBox ID="txEmail" runat="server" 
        style="z-index: 1; left: 10px; top: 44px; position: absolute; width: 308px"></asp:TextBox>
    <asp:TextBox ID="txSenha" runat="server" 
        
        
        style="z-index: 1; left: 9px; top: 115px; position: absolute; width: 307px" 
        TextMode="Password"></asp:TextBox>
    <asp:Button ID="btLogin" runat="server" onclick="btLogin_Click" 
        style="z-index: 1; left: 274px; top: 154px; position: absolute" Text="Login" />
    <asp:Label ID="lbSenha" runat="server" 
        style="z-index: 1; left: 10px; top: 87px; position: absolute" Text="Senha:"></asp:Label>
</asp:Content>
