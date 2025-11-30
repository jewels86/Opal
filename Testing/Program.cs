using Opal;

namespace Testing;

internal static class Program
{
    public static void Main()
    {
        using var context = new OpalContext(initializeInBackground: false, useGpu: true);
        
        AutogradTests.RunAll();
        //FfTests.OverfittingTest();
        //FfTests.OverfittingTestBatched();
    }
}