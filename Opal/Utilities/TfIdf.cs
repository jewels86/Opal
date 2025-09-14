using System.Collections.Concurrent;

namespace Opal.Utilities;

public class TfIdf
{
    public Dictionary<string, Dictionary<string, int>> DocumentWordCounts { get; } = [];
    public Dictionary<string, int> DocumentFrequencies { get; } = [];
    public int TotalDocuments { get; private set; }

    public void Add(string docName, string[] words)
    {
        DocumentWordCounts[docName] = [];
        HashSet<string> unique = [];

        foreach (string word in words)
        {
            DocumentWordCounts[docName][word] = DocumentWordCounts[docName].GetValueOrDefault(word, 0) + 1;
            unique.Add(word);
        }

        foreach (string word in unique)
        {
            DocumentFrequencies[word] = DocumentFrequencies.GetValueOrDefault(word, 0) + 1;
        }
        
        TotalDocuments++;
    }
    
    public void Remove(string docName)
    {
        if (!DocumentWordCounts.ContainsKey(docName))
            return;

        var words = DocumentWordCounts[docName].Keys.ToList();
        foreach (string word in words)
        {
            if (DocumentFrequencies.ContainsKey(word))
            {
                DocumentFrequencies[word]--;
                if (DocumentFrequencies[word] <= 0)
                    DocumentFrequencies.Remove(word);
            }
        }

        DocumentWordCounts.Remove(docName);
        TotalDocuments = Math.Max(0, TotalDocuments - 1);
    }
    
    public double GetTfIdf(string docName, string word)
    {
        if (!DocumentWordCounts.ContainsKey(docName) || !DocumentWordCounts[docName].ContainsKey(word))
            return 0.0;

        int wordCount = DocumentWordCounts[docName][word];
        int wordsInDoc = DocumentWordCounts[docName].Values.Sum();
        double tf = (double)wordCount / wordsInDoc;
        
        int docsWithWord = DocumentFrequencies.GetValueOrDefault(word, 0);
        double idf = Math.Log((double)TotalDocuments / (1 + docsWithWord));
        
        return tf * idf;
    }

    public double[] Salience(string[] sentence)
    {
        const string docName = "temp";
        Add(docName, sentence);
        double[] salience = sentence.Select(word => GetTfIdf(docName, word)).ToArray();
        Remove(docName);
        return salience;
    }
}

public class ConcurrentTfIdf
{
    public ConcurrentDictionary<string, ConcurrentDictionary<string, int>> DocumentWordCounts { get; } = new();
    public ConcurrentDictionary<string, int> DocumentFrequencies { get; } = new();
    private int _totalDocuments;
    public int TotalDocuments => _totalDocuments;

    public void Add(string docName, string[] words)
    {
        var wordCounts = new ConcurrentDictionary<string, int>();
        var unique = new ConcurrentBag<string>();
        words.AsParallel().ForAll(word =>
        {
            wordCounts.AddOrUpdate(word, 1, (_, v) => v + 1);
            unique.Add(word);
        });
        DocumentWordCounts[docName] = wordCounts;
        unique.Distinct().AsParallel().ForAll(word =>
        {
            DocumentFrequencies.AddOrUpdate(word, 1, (_, v) => v + 1);
        });
        Interlocked.Increment(ref _totalDocuments);
    }

    public void Remove(string docName)
    {
        if (!DocumentWordCounts.TryRemove(docName, out var wordCounts))
            return;
        wordCounts.Keys.AsParallel().ForAll(word =>
        {
            DocumentFrequencies.AddOrUpdate(word, 0, (_, v) => v > 1 ? v - 1 : 0);
            if (DocumentFrequencies[word] == 0)
                DocumentFrequencies.TryRemove(word, out _);
        });
        Interlocked.Decrement(ref _totalDocuments);
    }

    public double GetTfIdf(string docName, string word)
    {
        if (!DocumentWordCounts.TryGetValue(docName, out var wordCounts) || !wordCounts.TryGetValue(word, out int wordCount))
            return 0.0;
        int wordsInDoc = wordCounts.Values.Sum();
        double tf = (double)wordCount / wordsInDoc;
        int docsWithWord = DocumentFrequencies.GetOrAdd(word, 0);
        double idf = Math.Log((double)TotalDocuments / (1 + docsWithWord));
        return tf * idf;
    }

    public double[] Salience(string[] sentence)
    {
        const string docName = "temp";
        Add(docName, sentence);
        var salience = sentence.AsParallel().Select(word => GetTfIdf(docName, word)).ToArray();
        Remove(docName);
        return salience;
    }
}
