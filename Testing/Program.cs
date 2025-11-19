using System.Runtime.InteropServices;
using Opal;
using Opal.Autograd;

namespace Testing;

internal static class Program
{
    public static void Main()
    {
        for (int i = 0; i < 30; i++)
        {
            Operations.Controller.Return(Operations.AllocateBuffer(1));
            Operations.Controller.Return(Operations.AllocateBuffer(8));
        }
        Console.WriteLine($"GPU Available: {Operations.GpuAvailable}");
        FfTests.OverfittingTest();
        //DiagnosticTest.RunAll();
        //GpuTest.TestGpuBufferZeroing();
        //CatalogTests.RunAll();
        //AutogradTests.RunAll();
        //FfTests.RunAll();
        //RecurrentTests.RunAll();
        //Operations.GpuAvailable = true;
        //LstmTests.RunAll();
    }
}