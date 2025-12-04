using System.Collections.Concurrent;

namespace Jewels.Opal.Utilities.Concurrency;

public class ConcurrentHashSet<T> : ICollection<T> where T : notnull
{
    private readonly ConcurrentDictionary<T, byte> _dict = new();

    public bool Add(T item) => _dict.TryAdd(item, 0);
    public bool Remove(T item) => _dict.TryRemove(item, out _);
    public bool Contains(T item) => _dict.ContainsKey(item);
    public void Clear() => _dict.Clear();
    public int Count => _dict.Count;
    public bool IsReadOnly => false;

    void ICollection<T>.Add(T item) => Add(item);

    public IEnumerator<T> GetEnumerator() => _dict.Keys.GetEnumerator();
    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    public void CopyTo(T[] array, int arrayIndex) => _dict.Keys.CopyTo(array, arrayIndex);
}