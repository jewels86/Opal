using Opal;
using Opal.Modules;
using Opal.Modules.Patterns;
using Opal.Utilities;

namespace Testing;

public static class Utilities
{
    #region Reading Files and URLs
    public static string[] ReadFileLines(string filePath)
    {
        if (File.Exists(filePath))
        {
            return File.ReadAllLines(filePath);
        }
        else
        {
            Core.Log("testing-utilities", Logging.LogLevel.Info, $"File not found: {filePath} - returning empty array.");
            return [];
        }
    }
    public static string[] ReadUrlLines(string url)
    {
        try
        {
            using var client = new HttpClient();
            var content = client.GetStringAsync(url).Result;
            return content.Split(["\r\n", "\n"], StringSplitOptions.None);
        }
        catch (Exception ex)
        {
            Core.Log("testing-utilities", Logging.LogLevel.Error, $"Failed to download or read from URL: {url} - {ex.Message}");
            Core.Log("testing-utilities", Logging.LogLevel.Info, "Exception caught; returning empty array.");
            return [];
        }
    }
    #endregion
    #region Graphing

    public static void CreateLossesChart(List<double> losses, string path = "losses.png")
    {
        var xs = Graphing.SimpleXs(losses.Count);
        Graphing.Save(Graphing.Create([
            (xs, losses.ToArray(), "Losses"),
            (xs, Graphing.SimpleMovingAverage(losses.ToArray(), 10), "SMA(10) Losses"),
            (xs, Graphing.SimpleMovingAverage(losses.ToArray(), 100), "SMA(100) Losses"),
            (xs, Graphing.SimpleMovingAverage(losses.ToArray(), 1000), "SMA(1000) Losses")
        ], "Training Losses"), path, 800, 600);
    }
    #endregion
    #region Parsing and String Data Handling
    
    public static void TrainSemanticInterpreter(SemanticInterpreterModule semanticInterpreter, List<string> sentences)
    {
        foreach (string sentence in sentences)
        {
            string[] words = StringParsing.Parse(sentence);
            semanticInterpreter.Interpret(words);
        }
    }
    
    public static void ParallelTrainSemanticInterpreter(SemanticInterpreterModule semanticInterpreter, List<string> sentences)
    {
        foreach (string sentence in sentences)
        {
            string[] words = StringParsing.Parse(sentence);
            semanticInterpreter.ParallelInterpret(words);
        }
    }
    #endregion
}