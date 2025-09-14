using Opal.Utilities;
using Opal.Modules;
using Opal.Modules.Patterns;
using static Opal.Configurations.SemanticInterpreterConfigurations;
using Opal;
using Opal.Utilities.ANNs.Lstm;
using Opal.Utilities.ANNs.Recurrent;
using Spectre.Console;
using static Testing.Utilities;

namespace Testing;

public static class Program
{
    private const int CONTEXT_WINDOW_SIZE = 6;
    private const double SALIENCE_THRESHOLD = 0.3;
    
    public static void Main()
    {
        InitializeCore();
        
        // Load configuration and data
        var config = GetUserConfiguration();
        string[] sentences = GetAllSentences();
        var (stopwords, prefixes, suffixes) = LoadLanguageData();
        
        // Configure string parsing
        ConfigureStringParsing(stopwords, prefixes, suffixes);
        
        // Initialize modules
        var embeddings = new EmbeddingsModule<string>(64, 256, 256, 0.05, "word-embeddings");
        var semanticInterpreter = GenerateDefaultSemanticInterpreter(embeddings);
        var nextWordGenerator = new NextWordGenerationModule("next-word-generation", embeddings, semanticInterpreter, hiddenLayers: 3, batchSize: 16);
        
        // Handle embeddings
        HandleEmbeddings(config.CreateNewEmbeddings, embeddings, semanticInterpreter, sentences);
        
        // Handle salience model
        var salienceModel = HandleSalienceModel(config.CreateNewSalienceModel, sentences, embeddings);
        
        // Handle next word generation model
        HandleNextWordModel(config.CreateNewNextWordModel, nextWordGenerator, sentences);
        
        // Interactive loop
        RunInteractiveLoop(nextWordGenerator, salienceModel);
    }
    
    private static void InitializeCore()
    {
        Core.LogLevel = 0;
        Core.Initialize();
        
        AnsiConsole.Write(new FigletText("Opal Language Model")
            .Color(Color.Blue));
    }
    
    private static UserConfiguration GetUserConfiguration()
    {
        var panel = new Panel("Configuration Setup")
            .BorderColor(Color.Green);
        AnsiConsole.Write(panel);
        
        return new UserConfiguration
        {
            CreateNewEmbeddings = AnsiConsole.Prompt(new ConfirmationPrompt("Create new embeddings?")),
            CreateNewSalienceModel = AnsiConsole.Prompt(new ConfirmationPrompt("Train new salience detection model?")),
            CreateNewNextWordModel = AnsiConsole.Prompt(new ConfirmationPrompt("Train new next-word generation model?"))
        };
    }
    
    public static string[] GetAllSentences()
    {
        List<string> sentences = new();
        string[] urls = [
            "https://www.gutenberg.org/files/11/11-0.txt", // Alice in Wonderland
            "https://www.gutenberg.org/files/1342/1342-0.txt", // Pride and Prejudice
            "https://www.gutenberg.org/files/84/84-0.txt", // Frankenstein
            "https://www.gutenberg.org/files/74/74-0.txt", // Tom Sawyer
            "https://www.gutenberg.org/files/2701/2701-0.txt", // Moby Dick
            "https://www.gutenberg.org/files/1661/1661-0.txt", // Sherlock Holmes
            "https://www.gutenberg.org/files/345/345-0.txt", // Dracula
        ];

        AnsiConsole.Status()
            .Start("Loading corpus...", ctx =>
            {
                foreach (var url in urls)
                {
                    ctx.Status($"Loading {url}...");
                    sentences.AddRange(ReadUrlLines(url));
                }
            });
            
        Core.Log("program", Logging.LogLevel.Info, $"Loaded {sentences.Count} sentences from {urls.Length} sources");
        return sentences.ToArray();
    }
    
    private static (List<string> stopwords, List<string> prefixes, List<string> suffixes) LoadLanguageData()
    {
        var stopwords = ReadFileLines("stopwords.txt").ToList();
        var prefixes = new List<string>();
        var suffixes = new List<string>();
        
        foreach (var affix in ReadFileLines("affixes.txt"))
        {
            if (affix.StartsWith("-")) prefixes.Add(affix.Substring(1).Trim());
            else if (affix.EndsWith("-")) suffixes.Add(affix.Substring(0, affix.Length - 1).Trim());
        }
        
        Core.Log("program", Logging.LogLevel.Info, $"Loaded {stopwords.Count} stopwords, {prefixes.Count} prefixes, {suffixes.Count} suffixes");
        return (stopwords, prefixes, suffixes);
    }
    
    private static void ConfigureStringParsing(List<string> stopwords, List<string> prefixes, List<string> suffixes)
    {
        StringParsing.Stopwords = stopwords;
        StringParsing.Separators = StringParsing.StandardSeparators;
        StringParsing.Prefixes = prefixes;
        StringParsing.Suffixes = suffixes;
    }
    
    private static void HandleEmbeddings(bool createNew, EmbeddingsModule<string> embeddings, 
        SemanticInterpreterModule semanticInterpreter, string[] sentences)
    {
        if (createNew)
        {
            AnsiConsole.Status()
                .Start("Training embeddings...", ctx =>
                {
                    TrainSemanticInterpreter(semanticInterpreter, sentences.ToList());
                    embeddings.SaveEmbeddingsToFile("embeddings.bin");
                });
        }
        else
        {
            embeddings.LoadEmbeddingsFromFile("embeddings.bin");
            string[] words = embeddings.GetAllData().ToArray();
            semanticInterpreter.Added = new HashSet<string>(words);
            Core.Log("program", Logging.LogLevel.Info, $"Loaded {words.Length} word embeddings");
        }
    }
    
    private static LstmAttentionNetwork<LstmAttentionBackpropCache> HandleSalienceModel(bool createNew, string[] sentences, EmbeddingsModule<string> embeddings)
    {
        const string modelPath = "salience-model.bin";
        
        if (createNew)
        {
            return TrainSalienceModel(sentences, embeddings, modelPath);
        }
        
        // TODO: Load existing salience model
        Core.Log("program", Logging.LogLevel.Info, "Loading existing salience model...");
        return TrainSalienceModel(sentences, embeddings, modelPath); // Fallback for now
    }
    
    private static LstmAttentionNetwork<LstmAttentionBackpropCache> TrainSalienceModel(string[] sentences, 
        EmbeddingsModule<string> embeddings, string modelPath)
    {
        // Step 1: Generate TF-IDF training data
        var tfidf = GenerateTfIdfData(sentences);
        
        // Step 2: Create training examples for LSTM
        var trainingData = GenerateLstmSalienceTrainingData(sentences, tfidf, embeddings);
        
        // Step 3: Create and train LSTM attention model
        var salienceModel = CreateSalienceModel(embeddings.N);
        
        // Step 4: Train the model
        TrainSalienceModelWithData(salienceModel, trainingData);
        
        // TODO: Save model
        Core.Log("program", Logging.LogLevel.Info, $"Salience model training completed");
        
        return salienceModel;
    }
    
    private static TfIdf GenerateTfIdfData(string[] sentences)
    {
        var tfidf = new TfIdf();
        
        AnsiConsole.Status()
            .Start("Computing TF-IDF scores...", ctx =>
            {
                for (int i = 0; i < sentences.Length; i++)
                {
                    if (i % 1000 == 0) ctx.Status($"Processing sentence {i}/{sentences.Length}...");
                    
                    string[] words = StringParsing.Parse(sentences[i]);
                    if (words.Length > 0)
                    {
                        tfidf.Add($"doc_{i}", words);
                    }
                }
            });
            
        Core.Log("program", Logging.LogLevel.Info, "TF-IDF computation completed");
        return tfidf;
    }
    
    private static List<SalienceTrainingExample> GenerateLstmSalienceTrainingData(string[] sentences, 
        TfIdf tfidf, EmbeddingsModule<string> embeddings)
    {
        var trainingData = new List<SalienceTrainingExample>();
        
        AnsiConsole.Status()
            .Start("Generating LSTM training data...", ctx =>
            {
                foreach (var sentence in sentences)
                {
                    string[] words = StringParsing.Parse(sentence);
                    if (words.Length < 3) continue; // Skip very short sentences
                    
                    // Get TF-IDF salience scores
                    double[] tfidfScores = tfidf.Salience(words);
                    
                    // Convert to binary classification (you could also keep continuous)
                    double[] targets = tfidfScores.Select(score => score > SALIENCE_THRESHOLD ? 1.0 : 0.0).ToArray();
                    
                    // Convert words to embeddings
                    double[][] wordEmbeddings = words.Select(w => embeddings.GetEmbedding(w)?.Vector ?? new double[embeddings.N]).ToArray();
                    
                    trainingData.Add(new SalienceTrainingExample
                    {
                        WordEmbeddings = wordEmbeddings,
                        SalienceTargets = targets,
                        OriginalWords = words
                    });
                }
            });
            
        Core.Log("program", Logging.LogLevel.Info, $"Generated {trainingData.Count} salience training examples");
        return trainingData;
    }
    
    private static LstmAttentionNetwork<LstmAttentionBackpropCache> CreateSalienceModel(int embeddingSize)
    {
        // TODO: You'll need to adapt this to your actual LSTM attention architecture
        var network = new LstmAttentionNetwork<LstmAttentionBackpropCache>("salience-detector");
        
        // Add appropriate layers based on your architecture
        // This is a placeholder - you'll need to match your actual implementation
        
        return network;
    }
    
    private static void TrainSalienceModelWithData(LstmAttentionNetwork<LstmAttentionBackpropCache> model, 
        List<SalienceTrainingExample> trainingData)
    {
        int epochs = AnsiConsole.Prompt(new TextPrompt<int>("Enter epochs for salience training:").DefaultValue(50));
        double learningRate = AnsiConsole.Prompt(new TextPrompt<double>("Enter learning rate:").DefaultValue(0.01));
        
        AnsiConsole.Progress()
            .Start(ctx =>
            {
                var task = ctx.AddTask("[green]Training salience model[/]");
                
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    double totalLoss = 0.0;
                    
                    foreach (var example in trainingData)
                    {
                        // TODO: Implement actual training step
                        // This depends on your LSTM attention implementation
                        // You'll need to convert the training data to the format your network expects
                        
                        totalLoss += 0.1; // Placeholder
                    }
                    
                    double avgLoss = totalLoss / trainingData.Count;
                    Core.Log("training", Logging.LogLevel.Info, $"Epoch {epoch + 1}/{epochs}, Loss: {avgLoss:F4}");
                    
                    task.Value = ((double)(epoch + 1) / epochs) * 100;
                }
            });
    }
    
    private static void HandleNextWordModel(bool createNew, NextWordGenerationModule nextWordGenerator, string[] sentences)
    {
        const string modelPath = "next-word-generation.lstm.bin";
        
        if (createNew)
        {
            TrainNextWordModel(nextWordGenerator, sentences, modelPath);
        }
        else
        {
            try
            {
                nextWordGenerator.Lstm.From(LstmNetwork.Load(modelPath));
                Core.Log("program", Logging.LogLevel.Info, "Loaded existing next-word model");
            }
            catch
            {
                Core.Log("program", Logging.LogLevel.Warning, "Could not load model, training new one...");
                TrainNextWordModel(nextWordGenerator, sentences, modelPath);
            }
        }
    }
    
    private static void TrainNextWordModel(NextWordGenerationModule nextWordGenerator, string[] sentences, string modelPath)
    {
        int epochs = AnsiConsole.Prompt(new TextPrompt<int>("Enter epochs for next-word training:").DefaultValue(100));
        double learningRate = AnsiConsole.Prompt(new TextPrompt<double>("Enter learning rate:").DefaultValue(0.01));
        
        List<double> losses = new();
        
        AnsiConsole.Progress()
            .Start(ctx =>
            {
                var task = ctx.AddTask("[blue]Training next-word model[/]");
                int processedSentences = 0;
                
                foreach (string sentence in sentences)
                {
                    string[] words = StringParsing.Parse(sentence);
                    if (words.Length <= CONTEXT_WINDOW_SIZE) continue;
                    
                    for (int i = 0; i <= words.Length - CONTEXT_WINDOW_SIZE - 1; i++)
                    {
                        string[] inputSeq = words.Skip(i).Take(CONTEXT_WINDOW_SIZE).ToArray();
                        string[] targetSeq = [words[i + CONTEXT_WINDOW_SIZE]];
                        
                        var loss = nextWordGenerator.Train(inputSeq, targetSeq, epochs, learningRate);
                        losses.Add(loss.Average());
                    }
                    
                    processedSentences++;
                    task.Value = ((double)processedSentences / sentences.Length) * 100;
                }
            });
            
        nextWordGenerator.Lstm.Save(modelPath);
        CreateLossesChart(losses, "next-word-losses.png");
        Core.Log("program", Logging.LogLevel.Info, "Next-word model training completed");
    }
    
    private static void RunInteractiveLoop(NextWordGenerationModule nextWordGenerator, 
        LstmAttentionNetwork<LstmAttentionBackpropCache> salienceModel)
    {
        var panel = new Panel("Interactive Mode")
            .BorderColor(Color.Yellow);
        AnsiConsole.Write(panel);
        
        while (true)
        {
            var choice = AnsiConsole.Prompt(
                new SelectionPrompt<string>()
                    .Title("What would you like to do?")
                    .AddChoices([
                        "Generate text with special start",
                        "Generate text with specific length", 
                        "Generate text with special end",
                        "Test salience detection",
                        "Generate free text",
                        "Exit"
                    ]));
                    
            switch (choice)
            {
                case "Generate text with special start":
                    GenerateWithSpecialStart(nextWordGenerator);
                    break;
                case "Generate text with specific length":
                    GenerateWithLength(nextWordGenerator);
                    break;
                case "Generate text with special end":
                    GenerateWithSpecialEnd(nextWordGenerator);
                    break;
                case "Test salience detection":
                    TestSalienceDetection(salienceModel);
                    break;
                case "Generate free text":
                    GenerateFreeText(nextWordGenerator);
                    break;
                case "Exit":
                    Core.Log("program", Logging.LogLevel.Info, "Goodbye!");
                    return;
            }
            
            AnsiConsole.WriteLine();
        }
    }
    
    private static void GenerateWithSpecialStart(NextWordGenerationModule generator)
    {
        List<string> generated = ["special-start"];
        GenerateSequence(generator, generated, specialEnd: true);
    }
    
    private static void GenerateWithLength(NextWordGenerationModule generator)
    {
        string input = AnsiConsole.Prompt(new TextPrompt<string>("Enter starting text:"));
        int length = AnsiConsole.Prompt(new TextPrompt<int>("Enter desired length:").DefaultValue(10));
        
        string[] inputWords = StringParsing.Parse(input);
        List<string> generated = inputWords.ToList();
        GenerateSequence(generator, generated, maxLength: length);
    }
    
    private static void GenerateWithSpecialEnd(NextWordGenerationModule generator)
    {
        string input = AnsiConsole.Prompt(new TextPrompt<string>("Enter starting text:"));
        string[] inputWords = StringParsing.Parse(input);
        List<string> generated = inputWords.ToList();
        GenerateSequence(generator, generated, specialEnd: true);
    }
    
    private static void GenerateFreeText(NextWordGenerationModule generator)
    {
        string input = AnsiConsole.Prompt(new TextPrompt<string>("Enter starting text:"));
        string[] inputWords = StringParsing.Parse(input);
        List<string> generated = inputWords.ToList();
        GenerateSequence(generator, generated, maxLength: 20); // Default reasonable length
    }
    
    private static void GenerateSequence(NextWordGenerationModule generator, List<string> generated, 
        bool specialEnd = false, int maxLength = -1)
    {
        AnsiConsole.Status()
            .Start("Generating...", ctx =>
            {
                while (true)
                {
                    string newWord = generator.GenerateNext(generated.ToArray());
                    if (specialEnd && newWord == "[special-end]") break;
                    if (maxLength > 0 && generated.Count >= maxLength) break;
                    generated.Add(newWord);
                    
                    // Prevent infinite loops
                    if (generated.Count > 100) break;
                }
            });
            
        string result = string.Join(" ", generated);
        AnsiConsole.Write(new Panel(result)
            .Header("[green]Generated Text[/]")
            .BorderColor(Color.Green));
    }
    
    private static void TestSalienceDetection(LstmAttentionNetwork<LstmAttentionBackpropCache> salienceModel)
    {
        string input = AnsiConsole.Prompt(new TextPrompt<string>("Enter sentence to analyze:"));
        string[] words = StringParsing.Parse(input);
        
        // TODO: Use the salience model to predict salience scores
        // This is a placeholder implementation
        AnsiConsole.Write(new Panel("Salience detection not yet implemented")
            .Header("[yellow]Salience Analysis[/]")
            .BorderColor(Color.Yellow));
    }
}

public class UserConfiguration
{
    public bool CreateNewEmbeddings { get; set; }
    public bool CreateNewSalienceModel { get; set; }
    public bool CreateNewNextWordModel { get; set; }
}

public class SalienceTrainingExample
{
    public double[][] WordEmbeddings { get; set; } = [];
    public double[] SalienceTargets { get; set; } = [];
    public string[] OriginalWords { get; set; } = [];
}