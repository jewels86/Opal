namespace Opal.Utilities;

public static class Logging
{
    public enum LogLevel
    {
        LowDebug = 0,
        Debug = 1,
        HighDebug = 2,
        
        LowInfo = 3,
        Info = 4,
        HighInfo = 5,
        
        LowWarning = 6,
        Warning = 7,
        HighWarning = 8,
        
        LowError = 9,
        Error = 10,
        HighError = 11,
    }
    
    public static void StandardLog(string name, int level, string message)
    {
        if (level < Core.LogLevel) return;
        if (Core.LogWhitelist.Count != 0 && !Core.LogWhitelist.Contains(name)) return;
        if (Core.LogBlacklist.Contains(name)) return;
        Console.WriteLine($"[{name}] [{level}] {message}");
    }
}