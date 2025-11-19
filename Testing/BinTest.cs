using Opal.Autograd;

namespace Testing;

public static class BinTest
{
    public static void MessAround()
    {
        var a = Operations.NewVector([0, 2, 1]);
        var b = Operations.NewVector([1, 2, 3]);
        var c = Operations.Add(a, b);
    }
}