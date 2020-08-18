// Prof. André Gustavo Schaeffer
// UFFS - Campus Erechim(RS)
// Este programa foi criado para fins educativos e NÃO está protegido por nenhum direito 
// autoral. Em épocas de descrédito com relação à Educação Superior Pública de Qualidade, 
// cabe lembrar que estes vídeos e este material didático gratuito só são possíveis de serem 
// realizados e disponibilizados por termos (ainda), como docentes, o que chamamos de 
// Liberdade Acadêmica: para pensar e para produzir.
// Seja livre!

#include <SoftwareSerial.h>
SoftwareSerial BT(10, 11); //RX TX

void setup()
{
  Serial.begin(9600);
  BT.begin(9600);
  pinMode(12, OUTPUT);
}

int num, i=0;
char s[49];
void loop()
{
  while(!BT.available());
  if(BT.available()>0)
  {
    s[i] = BT.read();
    i++;
  }
  if(i==49)
  {
    
    // o laco abaixo potencializa quem esta marcado
    for(i=0; i<49; i++)
    {
      if(s[i]=='1')
      { 
        s[i]='4';
      }
    }

    // o laco abaixo aumenta de 0 para valor maior quem esta perto de um marcado
    for(i=0; i<49; i++)
    {
      if(s[i]=='4')
      { 
        if(i>6) 
        { if(s[i-7]=='0') s[i-7]='2'; }
        
        if((i!=0)&&(i!=7)&&(i!=14)&&(i!=21)&&(i!=28)&&(i!=35)&&(i!=42))
        { if(s[i-1]=='0') s[i-1]='2'; }
        
        if(i<42)
        { if(s[i+7]=='0') s[i+7]='2'; }
        
        if((i!=6)&&(i!=13)&&(i!=20)&&(i!=27)&&(i!=34)&&(i!=41)&&(i!=48))
        { if(s[i+1]=='0') s[i+1]='2'; }
        
      }
    }

   /* //gerar base de treinamento 
    Serial.print("base.addSample(("); 
    for(i=0; i<49; i++)
    {
      Serial.print(s[i]); if(i<48)Serial.print(","); 
    }
    Serial.println(" ),(0,0,0,0,0,0,0,0,0,0))"); 
   */
   
    Serial.flush(); 
    for(i=0; i<49; i++)
    {
      Serial.print(s[i]); if(i<48)Serial.print(",");
    }
    Serial.print("\n");
    delay(3000);  
    
  while(!Serial.available());
  if (Serial.available() > 0)
  {
   num=Serial.parseInt();
   if(num == 0) 
     for(i=0;i<20;i++) 
       {
        digitalWrite(12, HIGH); delay(100); digitalWrite(12, LOW); delay(100);
       }
       else
         for(i=0; i<num; i++)
           {
            digitalWrite(12, HIGH); delay(600); digitalWrite(12, LOW); delay(600);
           }
   } 
  
   i=0;
  }

  
}
