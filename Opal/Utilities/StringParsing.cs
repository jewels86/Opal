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
		public static List<string> Separators = [];

		public static string[] Parse(string input)
		{
			var words = input.Split(Separators.ToArray(), StringSplitOptions.RemoveEmptyEntries);

			var filteredWords = words
				.Where(word => !Stopwords.Contains(word.ToLower()))
				.Select(word => word.ToLower())
				.ToArray();
			return filteredWords;
		}
	}
}
