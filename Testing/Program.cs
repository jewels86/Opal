using System.Runtime.InteropServices;
using Jewels.Lazulite;
using Opal;

namespace Testing;

internal static class Program
{
    public static void Main()
    {
        //Operations.DefaultAcceleratorIndex = Compute.RequestAccelerator(false);
        //Operations.GpuAvailable = false;
        Console.WriteLine($"GPU Available: {Compute.Instance.GpuInUse}");
        //ScalarMultiplyDiagnosticTest.RunAll();
        //FfTests.OverfittingTest();
        //DiagnosticTest.RunAll();
        //GpuTest.TestGpuBufferZeroing();
        //CatalogTests.RunAll();
        //AutogradTests.RunAll();
        //AutogradTests.RunAll();
        FfTests.RunAll();
        //RecurrentTests.RunAll();
        //Operations.GpuAvailable = true;
        //LstmTests.RunAll();
        Operations.Dispose();
    }
}