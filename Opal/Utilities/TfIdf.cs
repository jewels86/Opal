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
        foreach (string word in unique) DocumentFrequencies[word] = DocumentFrequencies.GetValueOrDefault(word, 0) + 1;
        
        TotalDocuments++;
    }
    
    public void Remove(string docName)
    {
        if (!DocumentWordCounts.TryGetValue(docName, out Dictionary<string, int>? count))
            return;

        var words = count.Keys.ToList();
        foreach (string word in words)
        {
            if (!DocumentFrequencies.TryGetValue(word, out int value)) continue;
            DocumentFrequencies[word] = --value;
            if (DocumentFrequencies[word] <= 0)
                DocumentFrequencies.Remove(word);
        }

        DocumentWordCounts.Remove(docName);
        TotalDocuments = Math.Max(0, TotalDocuments - 1);
    }
    
    public double GetTfIdf(string docName, string word)
    {
        if (!DocumentWordCounts.TryGetValue(docName, out Dictionary<string, int>? value) || !value.TryGetValue(word, out int wordCount))
            return 0.0;

        int wordsInDoc = value.Values.Sum();
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