using Opal.Utilities;
using Opal.Modules;
using Opal.Modules.Patterns;
using static Opal.Configurations.SemanticInterpreterConfigurations;
using Opal;
using System.Net.Http;
using System.Net;

namespace Testing
{
	internal class Program
	{
		static void Main(string[] args)
		{
			List<string> sentences = new();

			if (File.Exists("data1.txt"))
			{
				//sentences.AddRange(File.ReadAllLines("data1.txt"));
			}
			if (File.Exists("data2.txt"))
			{
				//sentences.AddRange(File.ReadAllLines("data2.txt"));
			}
			if (File.Exists("data3.txt"))
			{
				//sentences.AddRange(File.ReadAllLines("data3.txt"));
			}

			// Download and read the text from the provided URL using HttpClient
			string url = "https://gist.githubusercontent.com/phillipj/4944029/raw/75ba2243dd5ec2875f629bf5d79f6c1e4b5a8b46/alice_in_wonderland.txt";
			try
			{
				using (var client = new HttpClient())
				{
					var aliceTextTask = client.GetStringAsync(url);
					aliceTextTask.Wait();
					string aliceText = aliceTextTask.Result;
					var aliceLines = aliceText.Split(new[] { "\r\n", "\n" }, StringSplitOptions.None);
					sentences.AddRange(aliceLines);
				}
			}
			catch (Exception ex)
			{
				Console.WriteLine($"Failed to download or read Alice text: {ex.Message}");
			}

			EmbeddingsModule<string> embeddings = new(64, 256, 256, 0.75, "word-embeddings");
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

			Core.LogLevel = 0;
			Core.Initialize();

			Core.Log("program", 1, $"Using stopwords {string.Join(", ", stopwordRecognition.Results())}");
			Core.Log("program", 1, $"Found suffixes: {string.Join(", ", suffixRecognition.Results().Select(x => new string(x.ToArray())))}");
			Core.Log("program", 1, $"Found prefixes: {string.Join(", ", prefixRecognition.Results().Select(x => new string(x.ToArray())))}");

			foreach (string sentence in sentences)
			{
				string[] words = StringParsing.Parse(sentence);
				semanticInterpreter.Interpret(words);
			}
			embeddings.SaveEmbeddingsToFile("embeddings.bin");
			
			int epochs = 20;
			double learningRate = 0.02;
			List<double> losses = new();

			foreach (string sentence in sentences)
			{
				string[] words = StringParsing.Parse(sentence);
				if (words.Length <= n) continue;
				Core.Log("program", (int)Logging.LogLevel.Info, $"Training with sentence: {sentence}");
				for (int i = 0; i <= words.Length - n - 1; i++)
				{
					string[] inputSeq = words.Skip(i).Take(n).ToArray();
					string[] targetSeq = new[] { words[i + n] };
					var loss = nextWordGeneration.Train(inputSeq, targetSeq, epochs, learningRate);
					losses.Add(loss.Average());
				}
			}
			nextWordGeneration.Lstm.Save("next-word-generation.lstm.bin");
			var xs = Graphing.SimpleXs(losses.Count);
            Graphing.Save(Graphing.Create([
	            (xs, losses.ToArray(), "Losses"),
	            (xs, Graphing.SimpleMovingAverage(losses.ToArray(), 10), "SMA(10) Losses"),
	            (xs, Graphing.SimpleMovingAverage(losses.ToArray(), 100), "SMA(100) Losses"),
	            (xs, Graphing.SimpleMovingAverage(losses.ToArray(), 1000), "SMA(1000) Losses")
            ], "Training Losses"), "losses.png", 800, 600);

			while (true)
			{
				string[] sentence;
				Console.Write("Generate a sentence? (y/nothing)");
				string? input = Console.ReadLine();
				if (input == "y")
				{
					Console.Write("Enter a starting sentence (or leave empty for special start): ");
					string inputSentence = Console.ReadLine() ?? string.Empty;

					sentence = string.IsNullOrWhiteSpace(inputSentence)
						? ["[special-start]"]
						: StringParsing.Parse(inputSentence);
					while (true)
					{
						if (sentence.Last() == "[special-end]")
						{
							break;
						}
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
