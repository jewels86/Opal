namespace Opal.Utilities.Concurrency;

public static class ConcurrentUtilities
{
    public static ParallelQuery<T> AsParallel<T>(this IEnumerable<T> source, bool parallel) =>
        parallel ? source.AsParallel() : source.AsParallel().WithDegreeOfParallelism(1);

    public static void ForAllParallel<T>(this IEnumerable<T> source, Action<T> action, bool parallel)
    {
        if (parallel)
            source.AsParallel().ForAll(action);
        else
            foreach (var item in source)
                action(item);
    }
}