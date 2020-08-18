using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace banco
{
    class BancoABC
    {
        private double appCli;

        public double AppCli
        {
            get { return appCli; }
            set { appCli = value; }
        }
        private double valIa;

        public double ValIa
        {
            get { return valIa; }
            set { valIa = value; }
        }
        private double ValAcm;

        public double ValAcm1
        {
            get { return ValAcm; }
            set { ValAcm = value; }
        }
        private double tsApl;

        public double TsApl
        {
            get { return tsApl; }
            set { tsApl = value; }
        }
        public double VsResg()
        {
        if ( AppCli == 1 )
        {
            return ((ValIa)+(ValIa*0.078))+((ValAcm*0.078)*TsApl);
        }
        if(AppCli = 2 && TsApl >= 24)
        {
            return ((ValIa)+(ValIa*0.078))+((ValAcm*0.097)*TsApl);
        }
        if(AppCli = 3 && TsApl = 6)
        {
            return ((ValIa)+(ValIa*0.085))+((ValAcm*0.097)*TsApl);
        }
        if(AppCli = 4 && TsApl >= 6 && TsApl < 12 )
        {
            return ((ValIa)+((ValAcm)*TsApl))/2;
        }
        if(AppCli = 5 && TsApl >= 12 )
        {
            return ((ValIa)+(ValIa*0.090))+((ValAcm*0.097)*TsApl);
        }
        else
        {
            return 0 ;
        }
        }
        
    }
}
