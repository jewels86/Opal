using System.Runtime.InteropServices;
using Opal;
using Opal.Autograd;

namespace Testing;

internal static class Program
{
    public static void Main()
    {
        Console.WriteLine($"GPU Available: {Operations.GpuAvailable}");
        //DiagnosticTest.RunAll();
        //GpuTest.TestGpuBufferZeroing();
        //CatalogTests.RunAll();
        //AutogradTests.RunAll();
        FfTests.RunAll();
        //RecurrentTests.RunAll();
        //Operations.GpuAvailable = true;
        //LstmTests.RunAll();
    }
}