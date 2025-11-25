using Jewels.Lazulite;
using Opal;

namespace Testing;

public static class AutogradTests
{
    public static void TestScalarGradients()
    {
        int aidx = Operations.DefaultAcceleratorIndex;
        Console.WriteLine("Testing Scalar Autograd...");
        
        // Build a simple graph: f(x,y,z) = (x * y) + z
        var x = Operations.New(2);
        var y = Operations.New(3);
        var z = Operations.New(4);
        
        var xy = Operations.Multiply(x, y);
        var result = Operations.Add(xy, z);
        
        // Forward: (2 * 3) + 4 = 10
        var resultValue = result.Value.ToHost();
        Console.WriteLine($"Forward pass: {resultValue} (expected 10.0)");
        Assert(Math.Abs(resultValue - 10.0) < 1e-10, "Forward pass failed");
        
        // Backward
        result.Backward(new ScalarValue(1, aidx));
        
        // Gradients: df/dx = y = 3, df/dy = x = 2, df/dz = 1
        var xGrad = x.Gradient.ToHost();
        var yGrad = y.Gradient.ToHost();
        var zGrad = z.Gradient.ToHost();
        
        Console.WriteLine($"dx: {xGrad} (expected 3.0)");
        Console.WriteLine($"dy: {yGrad} (expected 2.0)");
        Console.WriteLine($"dz: {zGrad} (expected 1.0)");
        
        Assert(Math.Abs(xGrad - 3.0) < 1e-10, "x gradient failed");
        Assert(Math.Abs(yGrad - 2.0) < 1e-10, "y gradient failed");
        Assert(Math.Abs(zGrad - 1.0) < 1e-10, "z gradient failed");
        
        Console.WriteLine("✓ Scalar test passed!\n");
    }
    
    public static void TestVectorGradients()
    {
        Console.WriteLine("Testing Vector Autograd...");
        
        // f(a,b) = dot(a, b) where a = [1,2], b = [3,4]
        var a = Operations.New([1, 2]);
        var b = Operations.New([3, 4]);
        
        var result = Operations.Dot(a, b);
        
        // Forward: 1*3 + 2*4 = 11
        var resultValue = result.Value.ToHost();
        Console.WriteLine($"Forward pass: {resultValue} (expected 11.0)");
        Assert(Math.Abs(resultValue - 11.0) < 1e-10, "Forward pass failed");
        
        // Backward
        result.Backward(new ScalarValue(1, a.AcceleratorIndex));
        
        // Gradients: df/da = b = [3,4], df/db = a = [1,2]
        var aGrad = a.Gradient.ToHost();
        var bGrad = b.Gradient.ToHost();
        
        Console.WriteLine($"da: [{aGrad[0]}, {aGrad[1]}] (expected [3.0, 4.0])");
        Console.WriteLine($"db: [{bGrad[0]}, {bGrad[1]}] (expected [1.0, 2.0])");
        
        Assert(Math.Abs(aGrad[0] - 3.0) < 1e-10, "a[0] gradient failed");
        Assert(Math.Abs(aGrad[1] - 4.0) < 1e-10, "a[1] gradient failed");
        Assert(Math.Abs(bGrad[0] - 1.0) < 1e-10, "b[0] gradient failed");
        Assert(Math.Abs(bGrad[1] - 2.0) < 1e-10, "b[1] gradient failed");
        
        Console.WriteLine("✓ Vector test passed!\n");
    }
    
    public static void TestComplexGraph()
    {
        Console.WriteLine("Testing Complex Graph...");
        
        // f(x,y) = (x + y) * (x - y)
        // At x=5, y=3: f = 8 * 2 = 16
        // df/dx = (x-y) + (x+y) = 2x = 10
        // df/dy = -(x+y) + (x-y) = -2y = -6
        
        var x = Operations.New(5);
        var y = Operations.New(3);
        
        var sum = Operations.Add(x, y);
        var diff = Operations.Subtract(x, y);
        var result = Operations.Multiply(sum, diff);
        
        var resultValue = result.Value.ToHost();
        Console.WriteLine($"Forward pass: {resultValue} (expected 16.0)");
        Assert(Math.Abs(resultValue - 16.0) < 1e-10, "Forward pass failed");
        
        result.Backward(new ScalarValue(1, x.AcceleratorIndex));
        
        var xGrad = x.Gradient.ToHost();
        var yGrad = y.Gradient.ToHost();
        
        Console.WriteLine($"dx: {xGrad} (expected 10.0)");
        Console.WriteLine($"dy: {yGrad} (expected -6.0)");
        
        Assert(Math.Abs(xGrad - 10.0) < 1e-10, "x gradient failed");
        Assert(Math.Abs(yGrad - (-6.0)) < 1e-10, "y gradient failed");
        
        Console.WriteLine("✓ Complex graph test passed!\n");
    }
    
    private static void Assert(bool condition, string message)
    {
        if (!condition)
            throw new Exception($"Assertion failed: {message}");
    }
    
    public static void RunAll()
    {
        TestScalarGradients();
        TestVectorGradients();
        TestComplexGraph();
        Console.WriteLine("All tests passed! ✓");
    }
}