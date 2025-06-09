using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace Opal.Modules.Patterns
{
    public class ExcessiveSubsequenceRecognitionModule<T> : IModule where T : notnull
    {
        public int ID { get; private set; }
        public string Name { get; private set; }
        public double K { get; set; }
        public int SubsequenceLength { get; set; }
        public Func<T, IEnumerable<T>>? SubsequenceExtractor { get; set; }

        public ConcurrentDictionary<T, int> SubsequenceCount { get; } = new();
        public ConcurrentBag<T> ExcessiveSubsequences { get; } = new();
        public int TotalSequences { get; private set; } = 0;

        private object _lock = new();
        public List<T[]> Sequences { get; } = new();

        public ExcessiveSubsequenceRecognitionModule(double k, int subsequenceLength, Func<T, IEnumerable<T>>? subsequenceExtractor = null, string? name = null)
        {
            ID = Core.Register(this);
            Name = name ?? $"excessive-subsequence-{typeof(T).Name.ToLower()}-{subsequenceLength}";
            K = k;
            SubsequenceLength = subsequenceLength;
            SubsequenceExtractor = subsequenceExtractor;
        }

        public void Initialize() { }

        public void Analyze(T[] sequence)
        {
            if (sequence.Length == 0) return;
            lock (_lock)
            {
                Sequences.Add(sequence);
            }
        }

        public void FinalizeAnalysis()
        {
            SubsequenceCount.Clear();
            ExcessiveSubsequences.Clear();
            TotalSequences = Sequences.Count;

            var subsequencesPerSequence = Sequences.Select(seq =>
                GetSubsequences(seq, SubsequenceLength).Distinct()
            );

            foreach (var subsequences in subsequencesPerSequence)
            {
                foreach (var subsequence in subsequences)
                {
                    SubsequenceCount.AddOrUpdate(subsequence, 1, (key, value) => value + 1);
                }
            }

            foreach (var kvp in SubsequenceCount)
            {
                if (kvp.Value >= TotalSequences * K)
                {
                    ExcessiveSubsequences.Add(kvp.Key);
                }
            }
        }

        public IEnumerable<T> GetExcessiveSubsequences()
        {
            return ExcessiveSubsequences.ToArray();
        }

        public bool IsExcessive(T subsequence)
        {
            return ExcessiveSubsequences.Contains(subsequence);
        }

        public void Clear()
        {
            SubsequenceCount.Clear();
            ExcessiveSubsequences.Clear();
            TotalSequences = 0;
            Sequences.Clear();
        }

        private IEnumerable<T> GetSubsequences(T[] sequence, int length)
        {
            if (sequence.Length < length) yield break;
            for (int i = 0; i <= sequence.Length - length; i++)
            {
                var subseq = sequence.Skip(i).Take(length).ToArray();
                if (SubsequenceExtractor != null)
                {
                    foreach (var s in SubsequenceExtractor((T)(object)subseq))
                        yield return s;
                }
                else
                {
                    yield return (T)(object)subseq;
                }
            }
        }
    }
}