using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal.Utilities
{
	public static class EmbeddingUtilities
	{
		public static double[] AverageVectors(double[][] vectors, int embeddingSize)
		{
			double[] average = new double[embeddingSize];
			for (int i = 0; i < embeddingSize; i++)
			{
				double sum = vectors.Sum(v => v[i]);
				average[i] = sum / vectors.Length;
			}
			return average;
		}

		public static double[] NormalizeVector(double[] vector)
		{
			double length = Math.Sqrt(vector.Sum(v => v * v));
			for (int i = 0; i < vector.Length; i++)
			{
				vector[i] /= length;
			}
			return vector;
		}

		public static double CosineSimilarity(double[] vectorA, double[] vectorB)
		{
			double dotProduct = 0;
			double lengthA = 0;
			double lengthB = 0;
			for (int i = 0; i < vectorA.Length; i++)
			{
				dotProduct += vectorA[i] * vectorB[i];
				lengthA += vectorA[i] * vectorA[i];
				lengthB += vectorB[i] * vectorB[i];
			}
			if (lengthA == 0 || lengthB == 0)
			{
				return 0;
			}
			return dotProduct / (Math.Sqrt(lengthA) * Math.Sqrt(lengthB));
		}
		public static double PearsonCorrelation(double[] vectorA, double[] vectorB)
		{
			double sumA = vectorA.Sum();
			double sumB = vectorB.Sum();
			double sumASquared = vectorA.Sum(v => v * v);
			double sumBSquared = vectorB.Sum(v => v * v);
			double sumProduct = 0;
			for (int i = 0; i < vectorA.Length; i++)
			{
				sumProduct += vectorA[i] * vectorB[i];
			}
			int n = vectorA.Length;
			double numerator = n * sumProduct - sumA * sumB;
			double denominator = Math.Sqrt((n * sumASquared - sumA * sumA) * (n * sumBSquared - sumB * sumB));
			if (denominator == 0)
			{
				return 0;
			}
			return numerator / denominator;
		}
		public static double EuclideanDistance(double[] vectorA, double[] vectorB)
		{
			double sum = 0;
			for (int i = 0; i < vectorA.Length; i++)
			{
				sum += (vectorA[i] - vectorB[i]) * (vectorA[i] - vectorB[i]);
			}
			return Math.Sqrt(sum);
		}
	}
}
