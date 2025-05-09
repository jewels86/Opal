using Opal.Utilities;
using Opal.Modules;
using Opal.Modules.Patterns;
using System.Numerics;
using static Opal.Configurations.SemanticInterpreterConfigurations;
using Opal;
using System.Transactions;

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

			EmbeddingsModule<string> embeddings = new(32, 256, 256, 0.5, "word-embeddings");
			SemanticInterpreterModule semanticInterpreter = new(
				GenerateNewStorageNodeWithEmbeddings(embeddings),
				GenerateRemoveStorageNodeWithEmbeddings(embeddings),
				GenerateGetSimilarityWithEmbeddings(embeddings),
				GenerateGetSimilarWordsWithEmbeddings(embeddings),
				GenerateAssociateWithEmbeddings(embeddings)
			);
			NextWordGenerationModule nextWordGeneration = new("next-word-generation", semanticInterpreter);
			ExcessiveUseRecognitionModule<string> stopwordRecognition = new(0.3, "stopword-recognition");
			ExcessiveUseRecognitionModule<string> wordStemFilter = new(0.5, "word-stem-filter");

			foreach (string sentence in sentences)
			{
				stopwordRecognition.Analyze(StringParsing.Split(sentence));
			}

			StringParsing.Stopwords = stopwordRecognition.ExcessiveTokens().ToList();
			StringParsing.Separators = StringParsing.StandardSeparators;

			Core.LogLevel = 2;
			Core.Initialize();

			Core.Log("program", 1, $"Using stopwords {string.Join(", ", StringParsing.Stopwords)}");

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
