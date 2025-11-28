using Opal;

namespace Testing;

internal static class Program
{
    public static void Main()
    {
        using var context = new OpalContext(initializeInBackground: true, useGpu: false);
        
        FfTests.OverfittingTest();
        FfTests.OverfittingTestBatched();
    }
}