-- --------------------------------------------------------
-- Servidor:                     localhost
-- Versão do servidor:           5.7.24 - MySQL Community Server (GPL)
-- OS do Servidor:               Win64
-- HeidiSQL Versão:              10.2.0.5599
-- --------------------------------------------------------

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8 */;
/*!50503 SET NAMES utf8mb4 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;


-- Copiando estrutura do banco de dados para cadastros_formulario
DROP DATABASE IF EXISTS `cadastros_formulario`;
CREATE DATABASE IF NOT EXISTS `cadastros_formulario` /*!40100 DEFAULT CHARACTER SET latin1 */;
USE `cadastros_formulario`;

-- Copiando estrutura para tabela cadastros_formulario.aposentpensio
DROP TABLE IF EXISTS `aposentpensio`;
CREATE TABLE IF NOT EXISTS `aposentpensio` (
  `IdAposentPensio` int(11) NOT NULL AUTO_INCREMENT,
  `Aposentado` tinyint(1) DEFAULT NULL,
  `Pensionista` tinyint(1) DEFAULT NULL,
  `NumBeneficio` varchar(20) DEFAULT NULL,
  `Periodo` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`IdAposentPensio`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Exportação de dados foi desmarcado.

-- Copiando estrutura para tabela cadastros_formulario.dadosprofissionais
DROP TABLE IF EXISTS `dadosprofissionais`;
CREATE TABLE IF NOT EXISTS `dadosprofissionais` (
  `IdDadosProfi` int(11) NOT NULL AUTO_INCREMENT,
  `Funcao` varchar(30) DEFAULT NULL,
  `DtAdmissao` date NOT NULL,
  `RendaMensal` double DEFAULT NULL,
  `OtRendas` double DEFAULT NULL,
  `TpRendas` varchar(40) DEFAULT NULL,
  `Patrimonio` double DEFAULT NULL,
  `TpCompRenda` int(11) DEFAULT NULL,
  `IdEmpresa` int(11) DEFAULT NULL,
  PRIMARY KEY (`IdDadosProfi`),
  KEY `fk_empresa` (`IdEmpresa`),
  CONSTRAINT `fk_empresa` FOREIGN KEY (`IdEmpresa`) REFERENCES `empresas` (`IdEmpresa`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Exportação de dados foi desmarcado.

-- Copiando estrutura para tabela cadastros_formulario.distribuidor
DROP TABLE IF EXISTS `distribuidor`;
CREATE TABLE IF NOT EXISTS `distribuidor` (
  `IdDistribuidor` int(11) NOT NULL AUTO_INCREMENT,
  `DA` varchar(50) NOT NULL,
  `FoneFixo` varchar(12) NOT NULL,
  `FoneCelular` varchar(12) NOT NULL,
  `Monitor` varchar(50) NOT NULL,
  `PreDistribu` varchar(50) NOT NULL,
  `MonitorPai` varchar(50) NOT NULL,
  `DataDis` date NOT NULL,
  PRIMARY KEY (`IdDistribuidor`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Exportação de dados foi desmarcado.

-- Copiando estrutura para tabela cadastros_formulario.empresas
DROP TABLE IF EXISTS `empresas`;
CREATE TABLE IF NOT EXISTS `empresas` (
  `IdEmpresa` int(11) NOT NULL AUTO_INCREMENT,
  `Nome` varchar(50) NOT NULL,
  `CNPJ` varchar(17) NOT NULL,
  `IdEndereco` int(11) DEFAULT NULL,
  PRIMARY KEY (`IdEmpresa`),
  KEY `fk_endereco_emp` (`IdEndereco`),
  CONSTRAINT `fk_endereco_emp` FOREIGN KEY (`IdEndereco`) REFERENCES `enderecos` (`IdEndereco`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Exportação de dados foi desmarcado.

-- Copiando estrutura para tabela cadastros_formulario.enderecos
DROP TABLE IF EXISTS `enderecos`;
CREATE TABLE IF NOT EXISTS `enderecos` (
  `IdEndereco` int(11) NOT NULL AUTO_INCREMENT,
  `endereco` varchar(100) NOT NULL,
  `Bairro` varchar(40) DEFAULT NULL,
  `Numero` varchar(10) DEFAULT NULL,
  `Complemento` varchar(30) DEFAULT NULL,
  `CEP` varchar(12) DEFAULT NULL,
  `Cidade` varchar(50) DEFAULT NULL,
  `UF` char(2) DEFAULT NULL,
  `TpResidencia` varchar(20) DEFAULT NULL,
  `TempResidencia` varchar(20) DEFAULT NULL,
  `VlPatrimonio` double DEFAULT NULL,
  PRIMARY KEY (`IdEndereco`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Exportação de dados foi desmarcado.

-- Copiando estrutura para tabela cadastros_formulario.financeiros
DROP TABLE IF EXISTS `financeiros`;
CREATE TABLE IF NOT EXISTS `financeiros` (
  `IdFinanceiro` int(11) NOT NULL AUTO_INCREMENT,
  `FormaPagamento` varchar(20) NOT NULL,
  `Parcelas` int(11) NOT NULL,
  `VlParcela` double NOT NULL,
  `VlFinanciado` double NOT NULL,
  `Venc1Parc` date NOT NULL,
  `Banco` varchar(5) NOT NULL,
  `Agencia` varchar(5) NOT NULL,
  `FoneAgen` varchar(12) NOT NULL,
  `ContCorrent` varchar(10) NOT NULL,
  `DtAbert` date NOT NULL,
  `NumCheque` varchar(25) NOT NULL,
  PRIMARY KEY (`IdFinanceiro`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Exportação de dados foi desmarcado.

-- Copiando estrutura para tabela cadastros_formulario.naturezaoperacao
DROP TABLE IF EXISTS `naturezaoperacao`;
CREATE TABLE IF NOT EXISTS `naturezaoperacao` (
  `IdOperacao` int(11) NOT NULL AUTO_INCREMENT,
  `TpOperacao` varchar(40) DEFAULT NULL,
  PRIMARY KEY (`IdOperacao`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Exportação de dados foi desmarcado.

-- Copiando estrutura para tabela cadastros_formulario.pessoas
DROP TABLE IF EXISTS `pessoas`;
CREATE TABLE IF NOT EXISTS `pessoas` (
  `IdPessoa` int(11) NOT NULL AUTO_INCREMENT,
  `Nome` varchar(50) NOT NULL,
  `DtNasc` date NOT NULL,
  `Nacionalidade` varchar(20) DEFAULT NULL,
  `CPF` double NOT NULL,
  `RG` double NOT NULL,
  `DtEmissao` date DEFAULT NULL,
  `OrgEmissor` char(3) DEFAULT NULL,
  `UF` char(2) DEFAULT NULL,
  `PaísDoc` varchar(25) DEFAULT NULL,
  `Naturalidade` varchar(25) DEFAULT NULL,
  `UFNat` varchar(2) DEFAULT NULL,
  `NomeMae` varchar(50) NOT NULL,
  `NomePai` varchar(50) NOT NULL,
  `IdNatOperacao` int(11) DEFAULT NULL,
  `IdEndereco` int(11) DEFAULT NULL,
  `IdDadosProfissionais` int(11) DEFAULT NULL,
  `IdRefPessoais` int(11) DEFAULT NULL,
  `IdAposentPensio` int(11) DEFAULT NULL,
  PRIMARY KEY (`IdPessoa`),
  UNIQUE KEY `CPF` (`CPF`),
  UNIQUE KEY `RG` (`RG`),
  KEY `fk_endereco` (`IdEndereco`),
  KEY `fk_dados_profissionais` (`IdDadosProfissionais`),
  KEY `fk_ref_pessoais` (`IdRefPessoais`),
  KEY `fk_aposent_pensio` (`IdAposentPensio`),
  KEY `fk_nat_operacao` (`IdNatOperacao`),
  CONSTRAINT `fk_aposent_pensio` FOREIGN KEY (`IdAposentPensio`) REFERENCES `aposentpensio` (`IdAposentPensio`),
  CONSTRAINT `fk_dados_profissionais` FOREIGN KEY (`IdDadosProfissionais`) REFERENCES `dadosprofissionais` (`IdDadosProfi`),
  CONSTRAINT `fk_endereco` FOREIGN KEY (`IdEndereco`) REFERENCES `enderecos` (`IdEndereco`),
  CONSTRAINT `fk_nat_operacao` FOREIGN KEY (`IdNatOperacao`) REFERENCES `naturezaoperacao` (`IdOperacao`),
  CONSTRAINT `fk_ref_pessoais` FOREIGN KEY (`IdRefPessoais`) REFERENCES `refpessoais` (`IdRefPesoal`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Exportação de dados foi desmarcado.

-- Copiando estrutura para tabela cadastros_formulario.produtos
DROP TABLE IF EXISTS `produtos`;
CREATE TABLE IF NOT EXISTS `produtos` (
  `IdProduto` int(11) NOT NULL AUTO_INCREMENT,
  `Nome` varchar(50) NOT NULL,
  PRIMARY KEY (`IdProduto`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Exportação de dados foi desmarcado.

-- Copiando estrutura para tabela cadastros_formulario.refpessoais
DROP TABLE IF EXISTS `refpessoais`;
CREATE TABLE IF NOT EXISTS `refpessoais` (
  `IdRefPesoal` int(11) NOT NULL AUTO_INCREMENT,
  `NomeSobrenome` varchar(20) DEFAULT NULL,
  `Afinidade` varchar(20) DEFAULT NULL,
  `FoneFixo` varchar(12) DEFAULT NULL,
  PRIMARY KEY (`IdRefPesoal`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Exportação de dados foi desmarcado.

/*!40101 SET SQL_MODE=IFNULL(@OLD_SQL_MODE, '') */;
/*!40014 SET FOREIGN_KEY_CHECKS=IF(@OLD_FOREIGN_KEY_CHECKS IS NULL, 1, @OLD_FOREIGN_KEY_CHECKS) */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
