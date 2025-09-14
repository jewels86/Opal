using ScottPlot;

namespace Opal.Utilities;

public static class Graphing
{
    public static Plot Create(List<(double[] x, double[] y, string? label)> series, string? title = null)
    {
        Plot plt = new();
        foreach ((double[] x, double[] y, string? label) in series)
        {
            var line = plt.Add.ScatterLine(x, y);
            if (label != null) line.LegendText = label;
        }

        plt.Title(title);
        return plt;
    }
    
    public static void Save(Plot plt, string path, int width = 800, int height = 600)
    {
        plt.SavePng(path, width, height);
    }

    public static double[] SimpleMovingAverage(double[] ys, int lookback)
    {
        double[] result = new double[ys.Length];
        for (int i = 0; i < ys.Length; i++)
        {
            if (i < lookback - 1)
            {
                result[i] = double.NaN;
            }
            else
            {
                double sum = 0;
                for (int j = i - lookback + 1; j <= i; j++)
                {
                    sum += ys[j];
                }
                result[i] = sum / lookback;
            }
        }
        return result;
    }

    public static double[] SimpleXs(int count)
    {
        double[] xs = new double[count];
        for (int i = 0; i < count; i++)
        {
            xs[i] = i;
        }
        return xs;
    }
}