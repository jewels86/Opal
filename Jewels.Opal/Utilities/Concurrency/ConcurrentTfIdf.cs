using System.Collections.Concurrent;

namespace Jewels.Opal.Utilities.Concurrency;

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
