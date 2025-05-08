using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal.Modules
{
	public class NextWordGenerationModule : IModule
	{
		public string Name { get; private set; }
		public int ID { get; private set; }

		/// <summary>The number of words to look behind to generate the next word.</summary>
		public int N { get; private set; } = 2;
		/// <summary>The weight a lookback word has on the next word generation.</summary>
		public double K { get; private set; } = 0.5;
		/// <summary>The number of next words to receive from the semantic interpreter.</summary>  
		public int M { get; private set; } = 5;
		/// <summary>The weight of a similar word on the next word generation.</summary>
		public double S { get; private set; } = 0.45;
		/// <summary>The weight of next word frequency on the next word generation.</summary>
		public double F { get; private set; } = 0.45;
		/// <summary>The randomness factor to inject into the next word generation.</summary>
		public double R { get; private set; } = 0.1;

		public SemanticInterpreterModule SemanticInterpreterModule { get; private set; }

		public NextWordGenerationModule(string? name = null, SemanticInterpreterModule? semanticInterpreterModule = null)
		{
			Name = name ?? "next-word-generation";
			ID = Core.Register(this);
			SemanticInterpreterModule = semanticInterpreterModule ?? new SemanticInterpreterModule();
		}

		public void Initialize() { }

		public string GenerateNext(string[] input)
		{
			List<(string, double)> contenders = [];
			if (input.Length == 0)
			{
				Core.Log(Name, 2, "Input is empty.");
				return string.Empty;
			}

			int last = input.Length - 1;
			for (int i = last - N; i < last; i++)
			{
				try
				{
					if (i < 0) continue;
					var similars = SemanticInterpreterModule.GetSimilar(input[i]);
					contenders.AddRange(similars.Select(x => (x.Item1, x.Item2 * K)));
				}
				catch (IndexOutOfRangeException)
				{
					Core.Log(Name, 3, "Index out of range: " + i);
					continue;
				}
			}

			var lastSimilars = SemanticInterpreterModule.GetSimilar(input[last]);
			contenders.AddRange(lastSimilars);

			var nextWords = SemanticInterpreterModule.NextWords(input[last], M);
			int total = nextWords.Select(x => x.Item2).Sum();
			List<(string, double)> probabilities = nextWords
				.Select(nextWords => (nextWords.Item1, (double)nextWords.Item2 / total))
				.ToList();

			var groupedContenders = contenders
				.GroupBy(x => x.Item1)
				.Select(g => (g.Key, g.Sum(x => x.Item2)))
				.ToDictionary(x => x.Key, x => x.Item2);

			var probDict = probabilities.ToDictionary(x => x.Item1, x => x.Item2);
			var allKeys = new HashSet<string>(groupedContenders.Keys.Concat(probDict.Keys));

			var final = allKeys.Select(word =>
			{
				double sim = groupedContenders.ContainsKey(word) ? groupedContenders[word] : 0.0;
				double freq = probDict.ContainsKey(word) ? probDict[word] : 0.0;
				double score = sim * S + freq * F;

				score += new Random().NextDouble() * R;

				return (word, score);
			})
			.OrderByDescending(x => x.score)
			.ToList();

			Core.Log(Name, 2, "Top candidate: " + final.FirstOrDefault().word);

			if (final.Count > 1)
			{
				double secondClosestScore = final[0].score - final[1].score;
				Core.Log(Name, 3, "Second closest candidate difference: " + secondClosestScore);
			}
			else
			{
				Core.Log(Name, 3, "No second candidate available.");
			}

			return final.FirstOrDefault().word;
		}
	}
}
