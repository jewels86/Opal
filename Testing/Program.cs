using Opal.Modules;
using Opal.Utilities;
using Opal.Utilities.ANNs.Lstm;
using Opal.Utilities.ANNs.Lstm.Attention;
using Spectre.Console;
using static Testing.Utilities;

namespace Testing;

internal static class Program
{
    static void Main()
    {
        AnsiConsole.MarkupLine("Welcome to the [green]Opal Testing Suite[/]!");
        string ans = AnsiConsole.Prompt(new SelectionPrompt<string>()
            .Title("What would you like to do?")
            .AddChoices([
                "(1) Train a new stopword recognition model",
                "(2) Train new word embeddings and a semantic interpreter",
                "(3) Load existing word embeddings and a semantic interpreter",
            ]));

        string[] sentences = GetAllSentences();

        if (ans.Contains("(1)"))
        {
            TrainStopwordModel(sentences);
        }
    }

    private static string[] GetAllSentences()
    {
        List<string> sentences = new();
        string[] urls = [
            "https://www.gutenberg.org/files/11/11-0.txt", // Alice in Wonderland (official)
            "https://www.gutenberg.org/files/1342/1342-0.txt", // Pride and Prejudice
            "https://www.gutenberg.org/files/84/84-0.txt", // Frankenstein
            "https://www.gutenberg.org/files/74/74-0.txt", // Tom Sawyer
            "https://www.gutenberg.org/files/2701/2701-0.txt", // Moby Dick
            "https://www.gutenberg.org/files/1661/1661-0.txt", // Sherlock Holmes
            "https://www.gutenberg.org/files/345/345-0.txt", // Dracula
        ];

        foreach (var url in urls) sentences.AddRange(ReadUrlLines(url));
        return sentences.ToArray();
    }

    private static void TrainStopwordModel(string[] sentences)
    {
        AnsiConsole.MarkupLine("Training stopword model; starting TF-IDF with the dataset...");
        ConcurrentTfIdf tfidf = new();
        //for (int i = 0; i < sentences.Length; i++) tfidf.Add(string.Intern($"doc_{i}"), StringParsing.Split(sentences[i]));
        sentences.AsParallel().ForAll(x => tfidf.Add(string.Intern($"doc_{Guid.NewGuid()}"), StringParsing.Split(x)));

        AnsiConsole.MarkupLine("TF-IDF will be computed on demand.");
        EmbeddingsModule<string> embeddings = GetEmbeddings();
        AnsiConsole.MarkupLine("Creating LSTM...");
        LstmDotAttentionNetwork lstm = new("stopword-lstm");

        int hiddenSize = AnsiConsole.Ask("What hidden size should the LSTM use?", 128);
        int layers = AnsiConsole.Ask("How many LSTM layers should be used?", 2);

        lstm.AddLayer(new LstmDotAttentionLayer(embeddings.N, hiddenSize, hiddenSize));
        for (int i = 0; i < layers; i++) lstm.AddLayer(new LstmDotAttentionLayer(hiddenSize, hiddenSize, hiddenSize));
        lstm.AddLayer(new LstmDotAttentionLayer(hiddenSize, hiddenSize, 1));

        // create sliding window sequences
        int n = AnsiConsole.Ask("What sliding window size should be used?", 5);
        List<List<double[]>> inputSequences = new();
        List<List<double[]>> outputSequences = new();
        List<List<double[]>> targetSequences = new();
        foreach (string sentence in sentences)
        {
            string[] words = StringParsing.Parse(sentence);
            if (words.Length <= n) continue;
            for (int i = 0; i <= words.Length - n - 1; i++)
            {
                string[] inputSeq = words.Skip(i).Take(n).ToArray();
                string[] targetSeq = [words[i + n]];
                inputSequences.Add(inputSeq.Select(w => embeddings.GetEmbedding(w)?.Vector ?? new double[embeddings.N])
                    .ToList());
                outputSequences.Add([]);
                double[] targetVec = new double[1];
                targetVec[0] = tfidf.GetTfIdf(string.Intern($"doc_{Guid.NewGuid()}"), targetSeq[0]);
                targetSequences.Add([targetVec]);
            }
        }

        AnsiConsole.MarkupLine($"Created {inputSequences.Count} training sequences.");
        int epochs = AnsiConsole.Ask("How many epochs should the training run for?", 10);
        double learningRate = AnsiConsole.Ask("What learning rate should be used?", 0.01);
        AnsiConsole.MarkupLine("Starting training...");
        List<double> losses = lstm.Train(inputSequences, outputSequences, targetSequences, epochs, learningRate);
        AnsiConsole.MarkupLine("Training complete!");
        string path = AnsiConsole.Ask("What path should the model be saved to?", "stopword-lstm.bin");
        lstm.Save(path);
        string chartPath = AnsiConsole.Ask("What path should the losses chart be saved to?", "losses.png");
        CreateLossesChart(losses, chartPath);
        AnsiConsole.MarkupLine("All done!");
    }

    private static EmbeddingsModule<string> GetEmbeddings()
    {
        string path = AnsiConsole.Ask("What is the path to the embeddings file? (if you have not created one, you should do so)", "embeddings.bin");
        EmbeddingsModule<string> embeddings = new(0, 0, 0, 0, "word-embeddings");
        embeddings.LoadEmbeddingsFromFile(path);
        return embeddings;
    }
}