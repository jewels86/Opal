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
			" ", "\n", "\t", ".", ",", ";", ":", "!", "?", "\"", "'", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\",
		];
	}
}
