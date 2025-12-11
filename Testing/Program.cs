using Jewels.Opal;

namespace Testing;

internal static class Program
{
    public static void Main()
    {
        using var context = new OpalContext(initializeInBackground: false, useGpu: false);
        
        //AutogradTests.RunAll();
        //FfTests.RunAll();
        Tests.SequenceMemoryTest();
        //RecurrentTests.RunAll();
    }
}