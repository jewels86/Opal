using Jewels.Opal;

namespace Testing;

internal static class Program
{
    public static void Main()
    {
        using var context = new OpalContext(initializeInBackground: false, useGpu: true);
        
        //AutogradTests.RunAll();
        //FfTests.BatchedTests();
        Tests.SequenceMemoryTest();
        //RecurrentTests.RunAll();
    }
}