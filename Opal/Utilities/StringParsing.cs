using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal.Utilities
{
	public static class StringParsing
	{
		public static List<string> Stopwords = [];
		public static List<string> Separators = [" ", "\n", "."];

		public static string[] Parse(string input)
		{
			var words = input.Split(Separators.ToArray(), StringSplitOptions.RemoveEmptyEntries);

			var filteredWords = words
				.Where(word => !Stopwords.Contains(word.ToLower()))
				.Select(word => word.ToLower())
				.ToArray();
			return filteredWords;
		}

		public static string[] Split(string input)
		{
			var words = input.Split(Separators.ToArray(), StringSplitOptions.RemoveEmptyEntries);
			return words;
		}

		public static List<string> StandardStopwords = [
			"the", "is", "in", "and", "to", "a", "of", "that", "it", "for",
			"on", "with", "as", "was", "at", "by", "an", "be", "this", "which",
			"or", "from", "but", "not", "are", "all", "if", "can", "we", "you",
		];
		public static List<string> StandardSeparators = [
			" ", "\n", "\t", ".", ",", ";", ":", "!", "?", "\"", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\"
		];
		
		public static string[] ExtractPrefixes(string sequence, int minLength, int maxLength)
		{
			List<char[]> results = new List<char[]>();
			char[] arr = sequence.ToCharArray();
			for (int i = 0; i < maxLength; i++)
			{
				if (i < minLength || i >= sequence.Length) continue;
				results.Add(sequence.Take(i).ToArray());
			}
			return results.Select(x => new string(x)).ToArray();
		}
		public static string[] ExtractSuffixes(string sequence, int minLength, int maxLength)
		{
			List<char[]> results = new List<char[]>();
			for (int i = 0; i < maxLength; i++)
			{
				if (i < minLength || i >= sequence.Length) continue;
				results.Add(sequence.Skip(i + 1).ToArray());
			}
			return results.Select(x => new string(x)).ToArray();
		}
		public static Func<string, string[]> PrefixExtractor(int minLength, int maxLength)
		{
			return sequence => ExtractPrefixes(sequence, minLength, maxLength);
		}
		public static Func<string, string[]> SuffixExtractor(int minLength, int maxLength)
		{
			return sequence => ExtractSuffixes(sequence, minLength, maxLength);
		}
	}
}
