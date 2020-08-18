#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "FreeRTOS.h"

int a=5;
int b=5;

void main(){
   if (a<100) if (b==5) printf("1"); else printf("2");
}