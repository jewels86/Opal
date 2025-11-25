using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal.Utilities
{
	public static class StringParsing
	{
		public static List<string> Stopwords { get; set; } = StandardStopwords;
		public static List<string> Separators { get; set; } = StandardSeparators;

		public static string[] Parse(string input)
		{
			var words = input.Split(Separators.ToArray(), StringSplitOptions.RemoveEmptyEntries);

			var filteredWords = words
				.Where(word => !Stopwords.Contains(word.ToLower()))
				.Select(word => word.ToLower())
				.Select(word => word.Trim())
				.Select(word =>
				{
					foreach (string prefix in Prefixes.Where(prefix => word.StartsWith(prefix))) word = word[prefix.Length..];
					foreach (string suffix in Suffixes.Where(suffix => word.EndsWith(suffix))) word = word[..^suffix.Length];
					return word;
				})
				.ToArray();
			return filteredWords;
		}

		public static string[] Split(string input)
		{
			var words = input.Split(Separators.ToArray(), StringSplitOptions.RemoveEmptyEntries);
			return words;
		}

		public static readonly List<string> StandardStopwords = [
			"the", "is", "in", "and", "to", "a", "of", "that", "it", "for",
			"on", "with", "as", "was", "at", "by", "an", "be", "this", "which",
			"or", "from", "but", "not", "are", "all", "if", "can", "we", "you",
		];
		public static readonly List<string> StandardSeparators = [
			" ", "\n", "\t", ".", ",", ";", ":", "!", "?", "\"", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\"
		];

		public static List<string> Prefixes { get; } = [];
		public static List<string> Suffixes { get; } = [];
	}
}
