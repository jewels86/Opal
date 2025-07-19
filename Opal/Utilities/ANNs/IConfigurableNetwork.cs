namespace Opal.Utilities.ANNs;

public interface IConfigurableNetwork : INueralNetwork
{
    public void AddLayer(ILayer layer);
}