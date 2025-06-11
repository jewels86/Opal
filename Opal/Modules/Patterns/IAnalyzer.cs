namespace Opal.Modules.Patterns;

public interface IAnalyzer<T> : IModule
{
    void Analyze(IEnumerable<T> sequence);
    void FinalizeAnalysis();
    IEnumerable<T> Results();
}