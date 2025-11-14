using Opal.Autograd;

public class AutogradTests
{
    public static void TestScalarGradients()
    {
        Console.WriteLine("Testing Scalar Autograd...");
        
        // Build a simple graph: f(x,y,z) = (x * y) + z
        var x = new Tensor<double>(2.0, null, _ => { }, 0.0);
        var y = new Tensor<double>(3.0, null, _ => { }, 0.0);
        var z = new Tensor<double>(4.0, null, _ => { }, 0.0);
        
        var xy = Operations.Multiply(new List<Tensor<double>> { x, y });
        var result = Operations.Sum(new List<Tensor<double>> { xy, z });
        
        // Forward: (2 * 3) + 4 = 10
        Console.WriteLine($"Forward pass: {result.Value} (expected 10.0)");
        Assert(Math.Abs(result.Value - 10.0) < 1e-10, "Forward pass failed");
        
        // Backward
        result.Backward(1.0);
        
        // Gradients: df/dx = y = 3, df/dy = x = 2, df/dz = 1
        Console.WriteLine($"dx: {x.Gradient} (expected 3.0)");
        Console.WriteLine($"dy: {y.Gradient} (expected 2.0)");
        Console.WriteLine($"dz: {z.Gradient} (expected 1.0)");
        
        Assert(Math.Abs(x.Gradient - 3.0) < 1e-10, "x gradient failed");
        Assert(Math.Abs(y.Gradient - 2.0) < 1e-10, "y gradient failed");
        Assert(Math.Abs(z.Gradient - 1.0) < 1e-10, "z gradient failed");
        
        Console.WriteLine("✓ Scalar test passed!\n");
    }
    
    public static void TestVectorGradients()
    {
        Console.WriteLine("Testing Vector Autograd...");
        
        // f(a,b) = dot(a, b) where a = [1,2], b = [3,4]
        var a = new Tensor<double[]>(new[] { 1.0, 2.0 }, null, _ => { }, new[] { 0.0, 0.0 });
        var b = new Tensor<double[]>(new[] { 3.0, 4.0 }, null, _ => { }, new[] { 0.0, 0.0 });
        
        var result = Operations.Dot(a, b);
        
        // Forward: 1*3 + 2*4 = 11
        Console.WriteLine($"Forward pass: {result.Value} (expected 11.0)");
        Assert(Math.Abs(result.Value - 11.0) < 1e-10, "Forward pass failed");
        
        // Backward
        result.Backward(1.0);
        
        // Gradients: df/da = b = [3,4], df/db = a = [1,2]
        Console.WriteLine($"da: [{a.Gradient[0]}, {a.Gradient[1]}] (expected [3.0, 4.0])");
        Console.WriteLine($"db: [{b.Gradient[0]}, {b.Gradient[1]}] (expected [1.0, 2.0])");
        
        Assert(Math.Abs(a.Gradient[0] - 3.0) < 1e-10, "a[0] gradient failed");
        Assert(Math.Abs(a.Gradient[1] - 4.0) < 1e-10, "a[1] gradient failed");
        Assert(Math.Abs(b.Gradient[0] - 1.0) < 1e-10, "b[0] gradient failed");
        Assert(Math.Abs(b.Gradient[1] - 2.0) < 1e-10, "b[1] gradient failed");
        
        Console.WriteLine("✓ Vector test passed!\n");
    }
    
    public static void TestComplexGraph()
    {
        Console.WriteLine("Testing Complex Graph...");
        
        // f(x,y) = (x + y) * (x - y)
        // At x=5, y=3: f = 8 * 2 = 16
        // df/dx = (x-y) + (x+y) = 2x = 10
        // df/dy = (x+y) - (x-y) = 2y = 6  -- wait, let me recalculate
        // f = (x+y)(x-y), df/dx = (x-y) + (x+y) = 2x = 10
        // df/dy = (x+y)*(-1) + (x-y)*(1) = -(x+y) + (x-y) = -2y = -6
        
        var x = new Tensor<double>(5.0, null, _ => { }, 0.0);
        var y = new Tensor<double>(3.0, null, _ => { }, 0.0);
        
        var sum = Operations.Sum(new List<Tensor<double>> { x, y });
        var diff = Operations.Subtract(x, y);
        var result = Operations.Multiply(new List<Tensor<double>> { sum, diff });
        
        Console.WriteLine($"Forward pass: {result.Value} (expected 16.0)");
        Assert(Math.Abs(result.Value - 16.0) < 1e-10, "Forward pass failed");
        
        result.Backward(1.0);
        
        Console.WriteLine($"dx: {x.Gradient} (expected 10.0)");
        Console.WriteLine($"dy: {y.Gradient} (expected -6.0)");
        
        Assert(Math.Abs(x.Gradient - 10.0) < 1e-10, "x gradient failed");
        Assert(Math.Abs(y.Gradient - (-6.0)) < 1e-10, "y gradient failed");
        
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