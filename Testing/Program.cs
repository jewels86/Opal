using Opal.Utilities;
using Opal.Modules;
using Opal.Modules.Patterns;
using static Opal.Configurations.SemanticInterpreterConfigurations;
using Opal;
using System.Net.Http;
using System.Net;
using Opal.Utilities.ANNs.Recurrent;
using Spectre.Console;
using static Testing.Utilities;

namespace Testing
{
	internal partial class Program
	{
		static void Main(string[] args)
		{
			List<string> sentences = new();
			string[] urls = [
				"https://gist.githubusercontent.com/phillipj/4944029/raw/75ba2243dd5ec2875f629bf5d79f6c1e4b5a8b46/alice_in_wonderland.txt"
			];

			//sentences.AddRange(ReadFileLines("data1.txt"));
			//sentences.AddRange(ReadFileLines("data2.txt"));
			//sentences.AddRange(ReadFileLines("data3.txt"));
			foreach (var url in urls) sentences.AddRange(ReadUrlLines(url));

			EmbeddingsModule<string> embeddings = new(64, 256, 256, 0.05, "word-embeddings");
			SemanticInterpreterModule semanticInterpreter = GenerateDefaultSemanticInterpreter(embeddings);
			NextWordGenerationModule nextWordGeneration = new("next-word-generation", embeddings, semanticInterpreter, hiddenLayers: 3, batchSize: 16);
			IrregularFrequencyRecognitionModule<string> stopwordRecognition = new(int.MaxValue, name: "stopword-recognition"); // TODO: stopword, prefix, and suffix recognition need to be replaced by a network
			ApproximateEqualityRecognitionModule<char> wordStemRecognition = new(0.8, x => x, name: "word-stem-recognition");
			IrregularFrequencyRecognitionModule<string> prefixRecognition = new(int.MaxValue, name: "prefix-recognition");
			IrregularFrequencyRecognitionModule<string> suffixRecognition =  new(int.MaxValue, name: "suffix-recognition");
			var prefixExtractor = StringParsing.PrefixExtractor(2, 5);
			var suffixExtractor = StringParsing.SuffixExtractor(2, 5);
			List<string> allWords = [];
			int n = 6;

			bool newEmbeddings = AnsiConsole.Prompt(new ConfirmationPrompt("Create new embeddings?"));
			
			foreach (string sentence in sentences)
			{
				string[] sequence = StringParsing.Split(sentence).Select(x => x.ToLower()).ToArray();
				stopwordRecognition.Analyze(sequence);
				allWords.AddRange(sequence);
			}
			stopwordRecognition.FinalizeAnalysis();

			AnalyzeSuffixesAndPrefixes(prefixRecognition, suffixRecognition, prefixExtractor, suffixExtractor, allWords);

			StringParsing.Stopwords = stopwordRecognition.Results().Distinct().ToList();
			StringParsing.Separators = StringParsing.StandardSeparators;
			
			Core.LogLevel = 0;
			Core.Initialize();

			Core.Log("program", Logging.LogLevel.Info, $"Using stopwords {string.Join(", ", stopwordRecognition.Results())}");
			Core.Log("program", Logging.LogLevel.Info, $"Found suffixes: {string.Join(", ", suffixRecognition.Results().Select(x => new string(x.ToArray())))}");
			Core.Log("program", Logging.LogLevel.Info, $"Found prefixes: {string.Join(", ", prefixRecognition.Results().Select(x => new string(x.ToArray())))}");

			if (newEmbeddings)
			{
				TrainSemanticInterpreter(semanticInterpreter, sentences);
				embeddings.SaveEmbeddingsToFile("embeddings.bin");
			}
			else
			{
				embeddings.LoadEmbeddingsFromFile("embeddings.bin");
				string[] words = embeddings.GetAllData().ToArray();
				semanticInterpreter.Added = new HashSet<string>(words);
			}
			
			bool newLstm = AnsiConsole.Prompt(new ConfirmationPrompt("Create new LSTM?"));

			if (newLstm)
			{
				int epochs = AnsiConsole.Prompt(new TextPrompt<int>("Enter number of epochs for training").DefaultValue(100));
				double learningRate = AnsiConsole.Prompt(new TextPrompt<double>("Enter learning rate for training").DefaultValue(0.01));
			
				List<double> losses = new();

				foreach (string sentence in sentences)
				{
					string[] words = StringParsing.Parse(sentence);
					if (words.Length <= n) continue;
					Core.Log("program", (int)Logging.LogLevel.Info, $"Training with sentence: {sentence}");
					for (int i = 0; i <= words.Length - n - 1; i++)
					{
						string[] inputSeq = words.Skip(i).Take(n).ToArray();
						string[] targetSeq = [words[i + n]];
						var loss = nextWordGeneration.Train(inputSeq, targetSeq, epochs, learningRate);
						losses.Add(loss.Average());
					}
				}
				
				string lstmPath = AnsiConsole.Prompt(
					new TextPrompt<string>("Enter path to save LSTM model (default: next-word-generation.lstm.bin):")
						.DefaultValue("next-word-generation.lstm.bin"));
				string lossesPath = AnsiConsole.Prompt(
					new TextPrompt<string>("Enter path to save losses chart (default: losses.png):")
						.DefaultValue("losses.png"));
				
				nextWordGeneration.Lstm.Save(lstmPath);
				CreateLossesChart(losses, lossesPath);
			}
			else
			{
				string lstmPath = AnsiConsole.Prompt(
					new TextPrompt<string>("Enter path to load LSTM model (default: next-word-generation.lstm.bin):")
						.DefaultValue("next-word-generation.lstm.bin"));
				nextWordGeneration.Lstm.From(LstmNetwork.Load(lstmPath));
			}

			while (true)
			{
				var choice = AnsiConsole.Prompt(
					new MultiSelectionPrompt<string>()
						.Title("What would you like to do?")
						.AddChoices([
							"[1] Generate a sentence with [special-start] and [special-end] tokens",
							"[2] Generate a sentence of a specific length with specific input",
							"[3] Generate a sentence with specific input and [special-end] token",
							"[E] Exit"
						])
				);
				
				if (choice.Contains("[E]"))
				{
					Core.Log("program", Logging.LogLevel.Info, "Exiting program.");
					break;
				}
				else
				{
					bool specialStart = choice.Contains("[1]");
					bool specialLength = choice.Contains("[2]");
					bool specialEnd = choice.Contains("[3]") || choice.Contains("[1]");
					string input = specialStart
						? "special-start"
						: AnsiConsole.Prompt(new TextPrompt<string>("Enter input sentence: "));
					int length = specialLength
						? AnsiConsole.Prompt(new TextPrompt<int>("Enter desired length of the sentence (default: 10):").DefaultValue(10))
						: -1;
					string[] inputWords = StringParsing.Parse(input);
					List<string> generated = inputWords.ToList();
					while (true)
					{
						string newWord = nextWordGeneration.GenerateNext(generated.ToArray());
						if (specialEnd && newWord == "[special-end]") break;
						generated.Add(newWord);
						if (specialLength && generated.Count >= length) break;
					}
					Core.Log("program", Logging.LogLevel.Info, $"Generated sentence: {string.Join(" ", generated)}");
				}
			}
		}
	}
}
