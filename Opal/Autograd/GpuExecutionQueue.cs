using ILGPU.Runtime;

namespace Opal.Autograd;

public class GpuExecutionQueue
{
    public bool AutoExecute { get; set; } = false;
    
    private readonly Accelerator _accelerator;
    private readonly List<Action> _operations = [];
    
    public GpuExecutionQueue(Accelerator accelerator) => _accelerator = accelerator;

    public void Enqueue(Action operation)
    {
        _operations.Add(operation);
        
        if (AutoExecute) operation();
    }

    public void Execute()
    {
        if (_operations.Count == 0) return;
        
        foreach (var operation in _operations) operation();
        _accelerator.Synchronize();
        _operations.Clear();
    }
}