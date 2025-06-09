using System;
using System.Collections.Generic;
using Opal.Modules.Patterns;

namespace Opal.Configurations
{
    public static class ExcessiveSubsequenceConfigurations
    {
        // For string: finds excessive substrings of specified length
        public static ExcessiveSubsequenceRecognitionModule<string> CreateForStringSubstrings(double k, int length, string? name = null)
        {
            return new ExcessiveSubsequenceRecognitionModule<string>(
                k,
                length,
                subseq => new[] { subseq },
                name
            );
        }

        // For char[]: finds excessive char n-grams (as string)
        public static ExcessiveSubsequenceRecognitionModule<string> CreateForCharNGrams(double k, int n, string? name = null)
        {
            return new ExcessiveSubsequenceRecognitionModule<string>(
                k,
                n,
                arr => new[] { new string((char[])(object)arr) },
                name
            );
        }

        // For int[]: finds excessive int n-grams (as int[])
        public static ExcessiveSubsequenceRecognitionModule<int[]> CreateForIntNGrams(double k, int n, string? name = null)
        {
            return new ExcessiveSubsequenceRecognitionModule<int[]>(
                k,
                n,
                subseq => new[] { subseq },
                name
            );
        }

        // For int[]: finds excessive int n-grams (as string)
        public static ExcessiveSubsequenceRecognitionModule<string> CreateForIntNGramStrings(double k, int n, string? name = null)
        {
            return new ExcessiveSubsequenceRecognitionModule<string>(
                k,
                n,
                arr => new[] { string.Join(",", (int[])(object)arr) },
                name
            );
        }

        // For T[]: finds excessive T n-grams (as T[])
        public static ExcessiveSubsequenceRecognitionModule<T[]> CreateForTNGrams<T>(double k, int n, string? name = null) where T : notnull
        {
            return new ExcessiveSubsequenceRecognitionModule<T[]>(
                k,
                n,
                subseq => new[] { subseq },
                name
            );
        }

        // For T[]: finds excessive T n-grams (as string)
        public static ExcessiveSubsequenceRecognitionModule<string> CreateForTNGramStrings<T>(double k, int n, Func<T, string> toStringFunc, string? name = null) where T : notnull
        {
            return new ExcessiveSubsequenceRecognitionModule<string>(
                k,
                n,
                arr => new[] { string.Join(",", ((T[])(object)arr).Select(toStringFunc)) },
                name
            );
        }

        // For generic T: pass a custom extractor
        public static ExcessiveSubsequenceRecognitionModule<T> CreateGeneric<T>(double k, int length, Func<T, IEnumerable<T>> extractor, string? name = null) where T : notnull
        {
            return new ExcessiveSubsequenceRecognitionModule<T>(k, length, extractor, name);
        }

        // For char[]: finds excessive prefixes of specified length (as string)
        public static ExcessiveSubsequenceRecognitionModule<string> CreateForCharPrefixes(double k, int n, string? name = null)
        {
            return new ExcessiveSubsequenceRecognitionModule<string>(
                k,
                n,
                arr => arr.Length >= n ? new[] { new string(((char[])(object)arr).Take(n).ToArray()) } : Array.Empty<string>(),
                name ?? $"excessive-prefix-{n}"
            );
        }

        // For char[]: finds excessive suffixes of specified length (as string)
        public static ExcessiveSubsequenceRecognitionModule<string> CreateForCharSuffixes(double k, int n, string? name = null)
        {
            return new ExcessiveSubsequenceRecognitionModule<string>(
                k,
                n,
                arr => arr.Length >= n ? new[] { new string(((char[])(object)arr).Skip(((char[])(object)arr).Length - n).ToArray()) } : Array.Empty<string>(),
                name ?? $"excessive-suffix-{n}"
            );
        }
    }
}
