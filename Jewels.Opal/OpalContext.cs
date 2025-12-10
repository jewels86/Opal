using Jewels.Lazulite;

namespace Jewels.Opal;

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
    
    public void BeginInitialization() => Compute.InitializeKernelsAsync();
    public void EnsureInitialization() => Compute.WaitForInitializationAsync();
    public void Initialize() => Compute.InitializeKernels(warmup: false);

    public void UseGpu(bool useGpu = true)
    {
        if (Operations.DefaultAcceleratorIndex != -1) Compute.ReleaseAccelerator(Operations.DefaultAcceleratorIndex);
        Operations.DefaultAcceleratorIndex = Compute.RequestAccelerator(useGpu);
    }
    
    public int GetAcceleratorIndex() => Operations.DefaultAcceleratorIndex;
    
    ~OpalContext() => Dispose();
    
    public void Dispose() => Operations.Dispose();
}