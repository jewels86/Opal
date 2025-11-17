using System.Runtime.InteropServices;
using Opal;
using Opal.Autograd;

namespace Testing;

internal static class Program
{
    public static void Main()
    {
        Console.WriteLine($"GPU Available: {Operations.GpuAvailable} (but will be switched off for testing)");
        Operations.GpuAvailable = false;
        CatalogTests.RunAll();
        AutogradTests.RunAll();
        FfTests.RunAll();
        RecurrentTests.RunAll();
        //LstmTests.RunAll();
    }
}