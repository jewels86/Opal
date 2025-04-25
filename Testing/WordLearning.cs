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

			//wordLearningManager.SentenceList.Add("The quick brown fox jumps over the lazy dog.");
			wordLearningManager.SentenceList.Add("Dogs are pets.");
			wordLearningManager.SentenceList.Add("Cats are pets.");
			wordLearningManager.SentenceList.Add("Dogs jump high.");
			//wordLearningManager.SentenceList.Add("Cats");

			stringParsing.Stopwords.Add("the");
			stringParsing.Stopwords.Add("is");
			stringParsing.Stopwords.Add("are");

			wordLearningManager.Initialize(ctx);
			lexicon.Initialize(ctx);
			stringParsing.Initialize(ctx);
			embeddingEngine.Initialize(ctx);
			semanticInterpreter.Initialize(ctx);

			ctx.Start();
		}
	}
}
