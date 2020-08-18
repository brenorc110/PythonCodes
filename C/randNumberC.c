#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <openssl/rand.h>
#include <time.h>

int main(void) {
  printf ("Resultado: %i", random_number(0, 50));
  return 0;
}

int random_number(int min_num, int max_num)
{
    int result = 0, low_num = 0, hi_num = 0;

    if (min_num < max_num)
    {
        low_num = min_num;
        hi_num = max_num + 1; // include max_num in output
    } else {
        low_num = max_num + 1; // include max_num in output
        hi_num = min_num;
    }
    
    srand(time(NULL));
    result = (rand() % (hi_num - low_num)) + low_num;
    return result;
}