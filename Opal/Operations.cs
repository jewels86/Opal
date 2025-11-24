using Jewels.Lazulite;

namespace Opal;

public static partial class Operations
{
    public static int DefaultAcceleratorIndex { get; }

    static Operations()
    {
        Compute.InitializeExtraKernels();
        DefaultAcceleratorIndex = Compute.RequestAccelerator();
    }
}