namespace Opal.Utilities.ANNs.Lstm;

using static MathFunctions;
using static Logging;
using static BinaryWriting;

public class LstmLayer<TInput, TOutput> : ILayer<TInput, TOutput> 
    where TInput : notnull where TOutput : notnull
{
    public int[] InputShape { get; }
    public int[] OutputShape { get; }
}