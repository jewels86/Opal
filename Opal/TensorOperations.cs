using Jewels.Lazulite;

namespace Opal;

public static partial class TensorOperations
{
    public static int DefaultAcceleratorIndex { get; }

    static TensorOperations()
    {
        Compute.InitializeExtraKernels();
        DefaultAcceleratorIndex = Compute.RequestAccelerator();
    }
}