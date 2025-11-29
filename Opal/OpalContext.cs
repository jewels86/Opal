using Jewels.Lazulite;

namespace Opal;

public sealed class OpalContext : IDisposable
{
    private static OpalContext? _instance;
    public static OpalContext GlobalContext => _instance ?? throw new Exception("OpalContext not initialized before use");

    public OpalContext(bool initializeInBackground = false, bool useGpu = true)
    {
        if (_instance != null) throw new Exception("OpalContext already initialized");
        _instance = this;
        
        if (initializeInBackground) BeginInitialization();
        else Initialize();
        UseGpu(useGpu);
    }
    
    public void BeginInitialization() => Operations.Compute.InitializeKernelsAsync();
    public void EnsureInitialization() => Operations.Compute.WaitForInitializationAsync();
    public void Initialize() => Operations.Compute.InitializeKernels(warmup: false);

    public void UseGpu(bool useGpu = true)
    {
        if (Operations.DefaultAcceleratorIndex != -1) Operations.Compute.ReleaseAccelerator(Operations.DefaultAcceleratorIndex);
        Operations.DefaultAcceleratorIndex = Operations.Compute.RequestAccelerator(useGpu);
    }
    
    public int GetAcceleratorIndex() => Operations.DefaultAcceleratorIndex;
    
    public void Dispose() => Operations.Dispose();

    ~OpalContext() => Dispose();
}