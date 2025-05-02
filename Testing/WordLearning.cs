using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Opal;
using Opal.Modules;

namespace Testing
{
	internal static class WordLearning
	{
		public static void Run(string[] args)
		{
			Context ctx = new();
			var wordLearningManager = new WordLearningManagerModule();
			var lexicon = new Opal.Modules.Strings.LexiconModule();
			var stringParsing = new Opal.Modules.Strings.StringParsingModule();
			var embeddingEngine = new Opal.Modules.Memory.EmbeddingEngineModule();
			var semanticInterpreter = new Opal.Modules.SemanticInterpreterModule();
			var nextWordGeneration = new Opal.Modules.Strings.NextWordGenerationModule();

			//wordLearningManager.SentenceList.Add("The quick brown fox jumps over the lazy dog.");  
			wordLearningManager.SentenceList.Add("Cats land on their feet.");
			wordLearningManager.SentenceList.Add("Dogs are pets.");
			wordLearningManager.SentenceList.Add("Cats are pets.");
			wordLearningManager.SentenceList.Add("Dogs can jump high.");
			wordLearningManager.SentenceList.Add("Cats can jump high.");
			wordLearningManager.SentenceList.Add("Cats eat food.");
			wordLearningManager.SentenceList.Add("Dogs eat food.");
			wordLearningManager.SentenceList.Add("Pets eat food.");
			wordLearningManager.SentenceList.Add("Birds can fly.");
			wordLearningManager.SentenceList.Add("Fish swim in water.");
			wordLearningManager.SentenceList.Add("Humans can think critically.");
			wordLearningManager.SentenceList.Add("Trees provide oxygen.");
			wordLearningManager.SentenceList.Add("The sun rises in the east.");
			wordLearningManager.SentenceList.Add("Rain falls from clouds.");
			wordLearningManager.SentenceList.Add("Fire is hot.");
			wordLearningManager.SentenceList.Add("Ice is cold.");
			wordLearningManager.SentenceList.Add("Books contain knowledge.");
			wordLearningManager.SentenceList.Add("Computers process data.");
			wordLearningManager.SentenceList.Add("Cars can drive fast.");
			wordLearningManager.SentenceList.Add("Planes fly in the sky.");
			wordLearningManager.SentenceList.Add("Ships sail on water.");
			wordLearningManager.SentenceList.Add("Children play games.");
			wordLearningManager.SentenceList.Add("Adults work in offices.");
			wordLearningManager.SentenceList.Add("Music soothes the soul.");
			wordLearningManager.SentenceList.Add("Art inspires creativity.");
			wordLearningManager.SentenceList.Add("Science explains the universe.");
			wordLearningManager.SentenceList.Add("Mathematics solves problems.");
			wordLearningManager.SentenceList.Add("History teaches lessons.");
			wordLearningManager.SentenceList.Add("Languages connect people.");

			stringParsing.Stopwords.Add("the");
			stringParsing.Stopwords.Add("is");
			stringParsing.Stopwords.Add("can");
			stringParsing.Stopwords.Add("are");

			wordLearningManager.Initialize(ctx);
			lexicon.Initialize(ctx);
			stringParsing.Initialize(ctx);
			embeddingEngine.Initialize(ctx);
			semanticInterpreter.Initialize(ctx);
			nextWordGeneration.Initialize(ctx);

			ctx.Start();
		}
	}
}
