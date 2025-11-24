using System.Runtime.InteropServices;
using Opal;
using Opal.Autograd;

namespace Testing;

internal static class Program
{
    public static void Main()
    {
        //Operations.GpuAvailable = false;
        Console.WriteLine($"GPU Available: {Operations.GpuAvailable}");
        //ScalarMultiplyDiagnosticTest.RunAll();
        //FfTests.OverfittingTest();
        //DiagnosticTest.RunAll();
        //GpuTest.TestGpuBufferZeroing();
        //CatalogTests.RunAll();
        //AutogradTests.RunAll();
        //FfTests.RunAll();
        //RecurrentTests.RunAll();
        Operations.GpuAvailable = true;
        LstmTests.RunAll();
    }
}