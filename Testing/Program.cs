using Opal.Utilities;
using Opal.Modules;
using Opal.Modules.Patterns;
using static Opal.Configurations.SemanticInterpreterConfigurations;
using static Opal.Configurations.ExcessiveSubsequenceConfigurations;
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
			SemanticInterpreterModule semanticInterpreter = new(
				GenerateNewStorageNodeWithEmbeddings(embeddings),
				GenerateRemoveStorageNodeWithEmbeddings(embeddings),
				GenerateGetSimilarityWithEmbeddings(embeddings),
				GenerateGetSimilarWordsWithEmbeddings(embeddings),
				GenerateAssociateWithEmbeddings(embeddings)
			);
			NextWordGenerationModule nextWordGeneration = new("next-word-generation", semanticInterpreter);
			ExcessiveUseRecognitionModule<string> stopwordRecognition = new(0.1, x => x.ToLower(), "stopword-recognition");
			ApproximateEqualityRecognitionModule<char> wordStemRecognition = new(0.8, x => x, name: "word-stem-recognition");
			ExcessiveSubsequenceRecognitionModule<string> prefixRecognition = CreateForCharPrefixes(0.2, 3, "prefix-recognition");
			ExcessiveSubsequenceRecognitionModule<string

			foreach (string sentence in sentences)
			{
				string[] sequence = StringParsing.Split(sentence);
				stopwordRecognition.Analyze(sequence);
				foreach (string word in sequence)
				{
					baseWordRecognition.Analyze(word.ToCharArray());
				}
			}
			stopwordRecognition.FinalizeAnalysis();
			baseWordRecognition.FinalizeAnalysis();

			StringParsing.Stopwords = stopwordRecognition.GetExcessiveTokens().Distinct().ToList();
			StringParsing.Separators = StringParsing.StandardSeparators;

			Core.LogLevel = 1;
			Core.Initialize();

			Core.Log("program", 1, $"Using stopwords {string.Join(", ", StringParsing.Stopwords)}");
			Core.Log("program", 1, $"Found suffixes and prefixes: {string.Join(", ", baseWordRecognition.GetExcessiveTokens())}")

			foreach (string sentence in sentences)
			{
				string[] words = StringParsing.Parse(sentence);
				semanticInterpreter.Interpret(words);
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
