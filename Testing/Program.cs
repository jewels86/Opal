using Opal.Utilities;
using Opal.Modules;
using Opal.Modules.Patterns;
using static Opal.Configurations.SemanticInterpreterConfigurations;
using Opal;

namespace Testing
{
	internal class Program
	{
		static void Main(string[] args)
		{
			List<string> sentences = new();

			if (File.Exists("data1.txt"))
			{
				sentences.AddRange(File.ReadAllLines("data1.txt"));
			}
			if (File.Exists("data2.txt"))
			{
				sentences.AddRange(File.ReadAllLines("data2.txt"));
			}
			if (File.Exists("data3.txt"))
			{
				sentences.AddRange(File.ReadAllLines("data3.txt"));
			}

			EmbeddingsModule<string> embeddings = new(32, 256, 256, 0.75, "word-embeddings");
			SemanticInterpreterModule semanticInterpreter = GenerateDefaultSemanticInterpreter(embeddings);
			NextWordGenerationModule nextWordGeneration = new("next-word-generation", embeddings, semanticInterpreter);
			IrregularFrequencyRecognitionModule<string> stopwordRecognition = new(3, name: "stopword-recognition");
			ApproximateEqualityRecognitionModule<char> wordStemRecognition = new(0.8, x => x, name: "word-stem-recognition");
			IrregularFrequencyRecognitionModule<string> prefixRecognition = new(3, name: "prefix-recognition");
			IrregularFrequencyRecognitionModule<string> suffixRecognition =  new(3, name: "suffix-recognition");
			var prefixExtractor = StringParsing.PrefixExtractor(2, 5);
			var suffixExtractor = StringParsing.SuffixExtractor(2, 5);
			List<string> allWords = [];
			int n = 4;
			
			foreach (string sentence in sentences)
			{
				string[] sequence = StringParsing.Split(sentence).Select(x => x.ToLower()).ToArray();
				stopwordRecognition.Analyze(sequence);
				allWords.AddRange(sequence);
			}
			stopwordRecognition.FinalizeAnalysis();

			foreach (string word in allWords.Distinct())
			{
				prefixRecognition.Analyze(prefixExtractor(word));
				suffixRecognition.Analyze(suffixExtractor(word));

			}
			prefixRecognition.FinalizeAnalysis();
			suffixRecognition.FinalizeAnalysis();

			StringParsing.Stopwords = stopwordRecognition.Results().Distinct().ToList();
			StringParsing.Separators = StringParsing.StandardSeparators;

			Core.LogLevel = 2;
			Core.Initialize();

			Core.Log("program", 1, $"Using stopwords {string.Join(", ", stopwordRecognition.Results())}");
			Core.Log("program", 1, $"Found suffixes: {string.Join(", ", suffixRecognition.Results().Select(x => new string(x.ToArray())))}");
			Core.Log("program", 1, $"Found prefixes: {string.Join(", ", prefixRecognition.Results().Select(x => new string(x.ToArray())))}");

			foreach (string sentence in sentences)
			{
				string[] words = StringParsing.Parse(sentence);
				semanticInterpreter.Interpret(words);
			}
			
			int epochs = 10;
			double learningRate = 0.01;

			foreach (string sentence in sentences)
			{
				string[] words = StringParsing.Parse(sentence);
				if (words.Length <= n) continue;
				for (int i = 0; i <= words.Length - n - 1; i++)
				{
					string[] inputSeq = words.Skip(i).Take(n).ToArray();
					string[] targetSeq = new[] { words[i + n] };
					nextWordGeneration.Train(inputSeq, targetSeq, epochs, learningRate);
				}
			}

			while (true)
			{
				string[] sentence;
				Console.Write("Generate a sentence? (y/nothing)");
				string? input = Console.ReadLine();
				if (input == "y")
				{
					Console.Write("Enter the max number of words: ");
					int maxWords = int.Parse(Console.ReadLine()!);

					sentence = new[] { "[special-start]" };
					while (sentence.Length < maxWords || sentence.Last() == "[special-end]")
					{
						string nextWord = nextWordGeneration.GenerateNext(sentence);
						sentence = sentence.Append(nextWord).ToArray();
					}
				}
				else
				{
					Console.WriteLine("Exiting...");
					break;
				}
				Console.WriteLine("Generated sentence: " + string.Join(" ", sentence));
			}
		}
	}
}
