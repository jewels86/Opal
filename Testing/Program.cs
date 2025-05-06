using Opal.Utilities;
using Opal.Modules;
using System.Numerics;
using static Opal.Configurations.SemanticInterpreterConfigurations;
using Opal;

namespace Testing
{
	internal class Program
	{
		static void Main(string[] args)
		{
			EmbeddingsModule<string> embeddings = new(32, 256, 256, 0.5, "word-embeddings");
			SemanticInterpreterModule semanticInterpreter = new(
				GenerateNewStorageNodeWithEmbeddings(embeddings),
				GenerateRemoveStorageNodeWithEmbeddings(embeddings),
				GenerateGetSimilarityWithEmbeddings(embeddings),
				GenerateGetSimilarWordsWithEmbeddings(embeddings),
				GenerateAssociateWithEmbeddings(embeddings)
			);
			StringParsing.Stopwords = StringParsing.StandardStopwords;
			StringParsing.Separators = StringParsing.StandardSeparators;

			List<string> sentences = [
				"The quick brown fox jumps over the lazy dog.",
				"Dogs are great companions.",
				"The fox is quick and clever.",
				"Companionship is important for dogs.",
				"The lazy dog sleeps all day.",
				"Foxes are wild animals.",
				"Quick thinking is a valuable trait.",
				"Lazy people often procrastinate.",
				"Companions provide emotional support.",
				"The dog barks at the fox.",
				"Quick brown foxes are fast.",
				"Lazy dogs enjoy lounging in the sun.",
				"Pets can be great companions.",
				"The fox and the dog are friends.",
				"Quick reflexes are essential for survival.",
				"Lazy afternoons are perfect for napping.",
				"Companionship can reduce stress.",
				"The dog chases the fox.",
				"Quickly, the fox escapes.",
				"Lazy days are meant for relaxation.",
				"Companions can be found in unexpected places.",
				"The fox is a cunning creature.",
				"Lazy dogs often snore.",
				"Quick decisions can lead to success.",
				"Companionship is a two-way street.",
				"The dog and the fox share a bond.",
				"Quick movements can startle a dog.",
				"Lazy mornings are best spent with a book.",
				"Companionship can be found in many forms.",
				"The fox is known for its agility.",
				"Lazy dogs love to play in the grass.",
				"Quick actions can prevent accidents.",
				"Companionship can bring joy.",
				"The dog and the fox have a unique relationship.",
				"Quickly, the dog learns new tricks.",
				"Lazy afternoons are perfect for a walk.",
				"Companionship can be a source of happiness.",
				"The fox is a symbol of cleverness.",
				"Lazy dogs enjoy belly rubs.",
				"Quick thinking can save lives.",
				"Companionship can be comforting.",
				"The dog and the fox play together.",
				"Quickly, the fox disappears into the woods.",
				"Lazy days are perfect for watching movies.",
				"Companionship can be found in friendships.",
				"The fox is a master of disguise.",
				"Lazy dogs love to chase their tails.",
				"Quick responses are appreciated in emergencies.",
				"Emergencies can be very bad.",
				"Quickly, the dog runs to its owner.",
				"The fire is dangerous.",
				"Quickly, the fox finds shelter.",
				"Lazy dogs no longer enjoy the shade.",
			];

			Core.Initialize();

			foreach (string sentence in sentences)
			{
				string[] words = StringParsing.Parse(sentence);
				semanticInterpreter.Interpret(words);
			}
		}
	}
}
