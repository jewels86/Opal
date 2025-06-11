namespace Opal.Modules.Patterns;

public class IrregularFrequencyRecognitionModule<T> : IModule, IAnalyzer<T>
{
    private readonly Dictionary<T, int> _counts = new();
    public List<T> IrregularTokens { get; private set; } = new();
    
    public string Name { get; private set; }
    public int ID { get; private set; }
    /// <summary>
    /// Threshold for irregularity detection.
    /// </summary>
    public float H { get; set; }
    
    public IrregularFrequencyRecognitionModule(float threshold = 2.0f, string name = "irregular-frequency-recognition")
    {
        ID = Core.Register(this);
        Name = name;
        H = threshold;
    }

    public void Analyze(IEnumerable<T> sequence)
    {
        foreach (var token in sequence)
        {
            if (_counts.ContainsKey(token))
                _counts[token]++;
            else
                _counts[token] = 1;
        }
    }

    public void FinalizeAnalysis()
    {
        var frequencies = _counts.Values.ToList();
        double mean = frequencies.Average();
        double stddev = Math.Sqrt(frequencies.Average(v => Math.Pow(v - mean, 2)));

        IrregularTokens = _counts
            .Where(kvp => Math.Abs(kvp.Value - mean) > H * stddev)
            .Select(kvp => kvp.Key)
            .ToList();
    }
    
    public IEnumerable<T> Results()
    {
        return IrregularTokens;
    }
}