using System;
using System.Linq;

namespace insertionCode
{
	class MainClass
	{


		static void Main(string[] args)
		{
			int[] v = new int[5];
			int i;

			for (i = 0; i < 5; i++)
			{
				Console.WriteLine("Entre com o número: ");
				v[i] = Convert.ToInt32(Console.ReadLine());
			}
			for (i = 0; i < 5; i++)
			{
				
				Console.WriteLine(i+1 + "a posicao do vetor em main: "+ v[i]);
			}

			for (i = 1; i < 5; i++)
			{
				int j;
				int aux = v[i];
				for (j = i; (j > 0) && (aux < v[j - 1]); j--)
					v[j] = v[j - 1];
				v[j] = aux;
			}

			for (i = 0; i < 5; i++)
			{
				Console.WriteLine(i+1 + "a posicao do vetor ordenado: "+ v[i]);
			}
			Console.ReadKey();
		}
	}
}
